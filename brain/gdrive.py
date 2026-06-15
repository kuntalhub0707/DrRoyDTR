"""
Google Drive integration.

Handles OAuth sign-in (browser popup), token persistence, dataset upload, and
validation of an existing Drive folder. All long-running calls are plain
functions so the page can run them on a background QThread (no UI freeze).

REQUIRES a one-time Google OAuth client file at:  DrRoyApp/client_secret.json
(create it free at console.cloud.google.com — see is_configured() / SETUP_HELP).
The user signs in through their own browser; this module never sees a password.
"""

import os
import re
import json
import time

from brain.paths import APP_ROOT
CLIENT_FILE = os.path.join(APP_ROOT, "client_secret.json")
TOKEN_FILE  = os.path.join(APP_ROOT, "gdrive_token.json")

# Full Drive scope so we can both create/upload folders AND read an existing
# folder the user pastes (Option B). userinfo gives us their display name.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

SETUP_HELP = (
    "Google sign-in needs a one-time free credential file.\n\n"
    "1. Go to console.cloud.google.com → create a project\n"
    "2. APIs & Services → Enable the 'Google Drive API'\n"
    "3. OAuth consent screen → External → add yourself as a test user\n"
    "4. Credentials → Create Credentials → OAuth client ID → Desktop app\n"
    "5. Download the JSON and save it as 'client_secret.json' in your DrRoyApp folder\n\n"
    "Then click Connect Google Drive again."
)


# ----------------------------------------------------------------------
# Configuration / connection state
# ----------------------------------------------------------------------
def is_configured():
    """True if the OAuth client file is present."""
    return os.path.isfile(CLIENT_FILE)


def install_client_secret(src):
    """
    Validate a user-picked OAuth client JSON and copy it into place as
    client_secret.json. Raises RuntimeError with a friendly message if the file
    is the wrong kind. Returns True on success.
    """
    import shutil
    try:
        with open(src, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        raise RuntimeError("That file isn't valid JSON. Please pick the file you downloaded from Google Cloud.")
    node = None
    if isinstance(data, dict):
        node = data.get("installed") or data.get("web")
    if not node or "client_id" not in node:
        raise RuntimeError(
            "That JSON isn't a Google OAuth client file.\n\nIn Google Cloud Console, create "
            "Credentials → OAuth client ID → type 'Desktop app', then download THAT file and pick it here.")
    shutil.copy(src, CLIENT_FILE)
    return True


def _load_token_creds():
    from google.oauth2.credentials import Credentials
    if not os.path.isfile(TOKEN_FILE):
        return None
    try:
        return Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    except Exception:
        return None


def is_connected():
    creds = _load_token_creds()
    return bool(creds and (creds.valid or creds.refresh_token))


def _save_token(creds, name=None):
    data = json.loads(creds.to_json())
    if name:
        data["_account_name"] = name
    with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def saved_account_name():
    if os.path.isfile(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh).get("_account_name", "")
        except Exception:
            return ""
    return ""


def disconnect():
    if os.path.isfile(TOKEN_FILE):
        try:
            os.remove(TOKEN_FILE)
        except Exception:
            pass


def _valid_credentials():
    """Return refreshed, valid credentials or raise a friendly error."""
    from google.auth.transport.requests import Request
    creds = _load_token_creds()
    if not creds:
        raise RuntimeError("Not connected to Google Drive yet.")
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds, saved_account_name())
    return creds


def _service(creds=None):
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds or _valid_credentials(), cache_discovery=False)


# ----------------------------------------------------------------------
# OAuth sign-in (opens the user's browser)
# ----------------------------------------------------------------------
def connect():
    """
    Run the OAuth flow: opens the system browser for Google sign-in + consent,
    captures the result on a localhost callback, saves the token, and returns
    the account's display name. Raises RuntimeError if not configured.
    """
    if not is_configured():
        raise RuntimeError("NOT_CONFIGURED")
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_FILE, SCOPES)
    creds = flow.run_local_server(
        port=0, open_browser=True,
        authorization_prompt_message="Opening your browser to sign in to Google…",
        success_message="Signed in. You can close this tab and return to Dr. Roy DT&R.")
    name = _fetch_account_name(creds)
    _save_token(creds, name)
    return name


def _fetch_account_name(creds):
    try:
        from googleapiclient.discovery import build
        info = build("oauth2", "v2", credentials=creds, cache_discovery=False).userinfo().get().execute()
        return info.get("name") or info.get("email") or "Google account"
    except Exception:
        try:
            about = _service(creds).about().get(fields="user(displayName,emailAddress)").execute()
            u = about.get("user", {})
            return u.get("displayName") or u.get("emailAddress") or "Google account"
        except Exception:
            return "Google account"


# ----------------------------------------------------------------------
# Local folder inspection
# ----------------------------------------------------------------------
def count_local_images(folder):
    """Return (num_images, num_subfolders_containing_images)."""
    n_images = 0
    sub_with_images = set()
    for dirpath, _dirs, files in os.walk(folder):
        imgs = [f for f in files if f.lower().endswith(IMG_EXTS)]
        if imgs:
            n_images += len(imgs)
            if os.path.normpath(dirpath) != os.path.normpath(folder):
                sub_with_images.add(os.path.normpath(dirpath))
    return n_images, len(sub_with_images)


# ----------------------------------------------------------------------
# Upload (Option A)
# ----------------------------------------------------------------------
def upload_folder(local_folder, on_progress=None, should_stop=None):
    """
    Upload every image in local_folder (preserving sub-folder structure) into a
    new Drive folder named DrRoyApp_Training_<date>. Returns (folder_id, name).
    on_progress(done, total, filename); should_stop() -> True to cancel.
    """
    from googleapiclient.http import MediaFileUpload
    svc = _service()

    top_name = "DrRoyApp_Training_" + time.strftime("%Y%m%d_%H%M")
    top = svc.files().create(
        body={"name": top_name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id").execute()
    top_id = top["id"]

    # gather images + their relative subdir
    items = []
    for dirpath, _dirs, files in os.walk(local_folder):
        rel = os.path.relpath(dirpath, local_folder)
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                items.append((os.path.join(dirpath, f), rel))
    total = len(items)

    folder_ids = {".": top_id}

    def ensure_subfolder(rel):
        if rel in folder_ids:
            return folder_ids[rel]
        parent = ensure_subfolder(os.path.dirname(rel)) if os.path.dirname(rel) not in ("", ".") else top_id
        meta = svc.files().create(
            body={"name": os.path.basename(rel), "mimeType": "application/vnd.google-apps.folder",
                  "parents": [parent]}, fields="id").execute()
        folder_ids[rel] = meta["id"]
        return meta["id"]

    done = 0
    for path, rel in items:
        if should_stop and should_stop():
            break
        parent_id = top_id if rel in (".", "") else ensure_subfolder(rel)
        media = MediaFileUpload(path, resumable=False)
        svc.files().create(body={"name": os.path.basename(path), "parents": [parent_id]},
                           media_body=media, fields="id").execute()
        done += 1
        if on_progress:
            on_progress(done, total, os.path.basename(path))
    return top_id, top_name


# ----------------------------------------------------------------------
# Validate an existing Drive folder (Option B)
# ----------------------------------------------------------------------
def parse_drive_id(link_or_id):
    """Extract a Drive folder ID from a share link or accept a raw ID."""
    s = (link_or_id or "").strip()
    if not s:
        return ""
    # /folders/<id>
    m = re.search(r"/folders/([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    # ?id=<id>  or  &id=<id>
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    # /d/<id>
    m = re.search(r"/d/([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    # raw id
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    return ""


def validate_folder(link_or_id):
    """
    Resolve a Drive folder link/ID and return (folder_id, name, image_count).
    Raises RuntimeError with a friendly message if invalid/not a folder.
    """
    fid = parse_drive_id(link_or_id)
    if not fid:
        raise RuntimeError("That doesn't look like a Google Drive folder link or ID.")
    svc = _service()
    try:
        meta = svc.files().get(fileId=fid, fields="id,name,mimeType",
                               supportsAllDrives=True).execute()
    except Exception:
        raise RuntimeError("Folder not found, or you don't have access to it.")
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise RuntimeError("That link points to a file, not a folder.")

    # count images inside (recursively, one level of paging)
    img_count = _count_drive_images(svc, fid)
    return fid, meta.get("name", "folder"), img_count


def colab_url(file_id):
    return f"https://colab.research.google.com/drive/{file_id}"


def upload_file_content(name, content_bytes, mime="application/json", parent_id=None):
    """Upload in-memory bytes as a Drive file. Returns the new file ID."""
    import io
    from googleapiclient.http import MediaIoBaseUpload
    svc = _service()
    body = {"name": name}
    if parent_id:
        body["parents"] = [parent_id]
    media = MediaIoBaseUpload(io.BytesIO(content_bytes), mimetype=mime, resumable=False)
    f = svc.files().create(body=body, media_body=media, fields="id").execute()
    return f["id"]


def get_file_text(file_id):
    """Return a small Drive file's contents as text."""
    svc = _service()
    data = svc.files().get_media(fileId=file_id).execute()
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)


def find_model_result(run_name, date):
    """
    Look in DrRoyApp_Models/<run>_<date>/ for best.pt. Returns a metadata dict
    {file_id, name, size, modified, metric, task} or None if not there yet.
    Reads DRROY_RESULT.json (written by the notebook) for the best accuracy.
    """
    svc = _service()
    folder = f"{run_name}_{date}"
    res = svc.files().list(
        q=f"name='{folder}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    folders = res.get("files", [])
    if not folders:
        return None
    fid = folders[0]["id"]
    r2 = svc.files().list(
        q=f"'{fid}' in parents and name='best.pt' and trashed=false",
        fields="files(id,name,size,modifiedTime)",
        supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    files = r2.get("files", [])
    if not files:
        return None
    best = files[0]
    meta = {"file_id": best["id"], "name": best["name"],
            "size": int(best.get("size", 0) or 0),
            "modified": best.get("modifiedTime", ""), "metric": None, "task": None}
    rj = svc.files().list(
        q=f"'{fid}' in parents and name='DRROY_RESULT.json' and trashed=false",
        fields="files(id)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get("files", [])
    if rj:
        try:
            d = json.loads(get_file_text(rj[0]["id"]))
            meta["metric"] = d.get("metric")
            meta["task"] = d.get("task")
        except Exception:
            pass
    return meta


# kept for compatibility: returns just the best.pt file id
def find_model_file(run_name, date):
    meta = find_model_result(run_name, date)
    return meta["file_id"] if meta else None


def download_file(file_id, dest, on_progress=None):
    """Download a Drive file by ID to a local path, reporting integer percent."""
    from googleapiclient.http import MediaIoBaseDownload
    svc = _service()
    req = svc.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = dl.next_chunk()
            if on_progress and status:
                on_progress(int(status.progress() * 100))
    if on_progress:
        on_progress(100)
    return dest


def _count_drive_images(svc, folder_id, depth=0):
    if depth > 6:
        return 0
    count = 0
    page = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType)",
            pageSize=1000, pageToken=page,
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                count += _count_drive_images(svc, f["id"], depth + 1)
            elif f["name"].lower().endswith(IMG_EXTS):
                count += 1
        page = resp.get("nextPageToken")
        if not page:
            break
    return count

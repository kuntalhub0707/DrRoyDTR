; Inno Setup script for Dr. Roy Data Training & Reporting
; Packages the self-contained bundle (Python + libraries + app) into a per-user
; installer with Start Menu + Desktop shortcuts and a Windows Settings uninstall entry.

#define MyAppName "Dr. Roy Data Training & Reporting"
#define MyAppShortName "DrRoyDTR"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "Dr. Kuntal Roy"

[Setup]
AppId={{A7E3C1B2-9F4D-4E6A-8B1C-DR0ROYDTR1000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription=AI-powered histopathology and hematology training and reporting tool
DefaultDirName={localappdata}\Programs\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer
OutputBaseFilename=DrRoyDTR_Setup_{#MyAppVersion}
SetupIconFile=DrRoyDTR.ico
UninstallDisplayIcon={app}\app\DrRoyDTR.ico
UninstallDisplayName={#MyAppName}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DirExistsWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; the whole self-contained bundle: python\ (bundled Python + libs) and app\
Source: "package\DrRoyDTR\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\DrRoyDTR.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\DrRoyDTR.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\main.py"""; WorkingDir: "{app}\app"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent

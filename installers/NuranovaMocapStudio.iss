; Inno Setup 6 Script for Nuranova Mocap Studio v1.0.0
; Developer: Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)

#define MyAppName "Nuranova Mocap Studio"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Nuranova Solutions"
#define MyAppURL "https://nuranova.lk/"
#define MyAppExeName "NuranovaMocapStudio.exe"

[Setup]
AppId={{7E8B4A12-3D5C-4A9F-B21C-89C4E12B1000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Nuranova Solutions\Nuranova Mocap Studio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\licenses\THIRD_PARTY_NOTICES.txt
OutputDir=output
OutputBaseFilename=NuranovaMocapStudio_Setup
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "firewallrule"; Description: "Allow Nuranova Mocap Studio local LAN communication (TCP port 8765, requires admin if elevated)"; GroupDescription: "Local Network"; Flags: unchecked

[Files]
Source: "..\dist\NuranovaMocapStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "NuranovaMocapBridge.zip"; DestDir: "{app}\blender_extension"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\documentation\printable\*"; DestDir: "{app}\printable_templates"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\USER_GUIDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\HARDWARE_TEST_GUIDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\LICENSE_REGISTER.md"; DestDir: "{app}\docs"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\User Guide"; Filename: "{app}\docs\USER_GUIDE.md"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""Nuranova Mocap Studio LAN"" dir=in action=allow protocol=TCP localport=8765 profile=private"; Flags: runhidden; Tasks: firewallrule
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

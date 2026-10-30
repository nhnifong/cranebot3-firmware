# Install this source tree's fan_control.py on every component (anchors and
# gripper) listed in a robot config, and restart the fan-control service.
# Asks for the pi user's password once and reuses it for every component.
#
#   powershell -ExecutionPolicy Bypass -File scripts\update_fan_control.ps1 configuration.json
#
# Uses the OpenSSH client built into Windows 10/11. ssh has no option to take a
# password, so it is handed over through SSH_ASKPASS: ssh runs a small helper
# that prints the password from an environment variable.
param(
    [Parameter(Mandatory = $true)][string]$Config
)
$ErrorActionPreference = 'Stop'

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error 'ssh not found: install the OpenSSH Client optional feature'
}

$repo = Split-Path -Parent $PSScriptRoot
$src = Join-Path $repo 'stringman-pilot-rpi-image\fan_control.py'

$conf = Get-Content -Raw $Config | ConvertFrom-Json
$addresses = @(@($conf.anchors) + @($conf.gripper) | Where-Object { $_.address } | ForEach-Object { $_.address })
if ($addresses.Count -eq 0) { Write-Error "no component addresses in $Config" }
Write-Host "components: $($addresses -join ' ')"

$secure = Read-Host 'password for pi' -AsSecureString
$password = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))

$tmp = Join-Path $env:TEMP "update_fan_control_$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    # A Windows checkout may have CRLF line endings, which break the script's
    # shebang on the Pi, so send a copy with LF endings.
    $script = Join-Path $tmp 'fan_control.py'
    $text = [IO.File]::ReadAllText($src) -replace "`r`n", "`n"
    [IO.File]::WriteAllText($script, $text, (New-Object Text.UTF8Encoding $false))

    $askpass = Join-Path $tmp 'askpass.cmd'
    Set-Content -Encoding ASCII $askpass '@powershell -NoProfile -Command "[Console]::Out.Write($env:STRINGMAN_SSH_PASS)"'

    $env:STRINGMAN_SSH_PASS = $password
    $env:SSH_ASKPASS = $askpass
    $env:SSH_ASKPASS_REQUIRE = 'force'  # OpenSSH 8.4+
    $setDisplay = -not $env:DISPLAY
    if ($setDisplay) { $env:DISPLAY = 'none' }  # older OpenSSH only uses askpass with DISPLAY set

    $sshOpts = @('-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=5')
    $remote = 'sudo install -m 755 /tmp/fan_control.py /usr/local/bin/fan_control.py && ' +
              'sudo systemctl restart fan-control && sleep 3 && ' +
              'journalctl -u fan-control -n 2 --no-pager -o cat && pinctrl get 12,18'

    $failed = @()
    foreach ($addr in $addresses) {
        Write-Host "== $addr"
        scp @sshOpts $script "pi@${addr}:/tmp/fan_control.py"
        if ($LASTEXITCODE -eq 0) {
            ssh @sshOpts "pi@$addr" $remote
        }
        if ($LASTEXITCODE -ne 0) { $failed += $addr }
    }
}
finally {
    Remove-Item Env:STRINGMAN_SSH_PASS, Env:SSH_ASKPASS, Env:SSH_ASKPASS_REQUIRE -ErrorAction SilentlyContinue
    if ($setDisplay) { Remove-Item Env:DISPLAY -ErrorAction SilentlyContinue }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

if ($failed.Count -gt 0) {
    Write-Host "FAILED: $($failed -join ' ')" -ForegroundColor Red
    exit 1
}
Write-Host 'all components updated'

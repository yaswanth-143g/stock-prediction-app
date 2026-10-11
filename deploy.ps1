# PowerShell 1-Click Remote Cloud Deployment Script
# Enterprise NSE Stock Price Prediction & Quantitative Portfolio Platform

param (
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,

    [Parameter(Mandatory=$false)]
    [string]$SSHUser = "root",

    [Parameter(Mandatory=$false)]
    [string]$SSHKeyPath = ""
)

Write-Host "🐳 Deploying NSE Stock Platform to Cloud Server ($ServerIP)..." -ForegroundColor Cyan

$SSHCommand = "ssh"
if ($SSHKeyPath -ne "") {
    $SSHCommand = "ssh -i '$SSHKeyPath'"
}

Write-Host "📦 Copying files to server..." -ForegroundColor Yellow
$SCPCommand = "scp"
if ($SSHKeyPath -ne "") {
    $SCPCommand = "scp -i '$SSHKeyPath'"
}

Invoke-Expression "$SCPCommand -r ./* ${SSHUser}@${ServerIP}:~/app/"

Write-Host "🏗️ Executing remote Docker deployment script..." -ForegroundColor Yellow
Invoke-Expression "$SSHCommand ${SSHUser}@${ServerIP} 'cd ~/app && chmod +x deploy.sh && ./deploy.sh'"

Write-Host "====================================================================" -ForegroundColor Green
Write-Host "✅ REMOTE CLOUD DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host "🌐 Access your live 24/7 trading cockpit at: http://${ServerIP}" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green

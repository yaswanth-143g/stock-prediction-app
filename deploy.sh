#!/bin/bash
# ==============================================================================
# 1-Click AWS EC2 / DigitalOcean Deployment Script
# Enterprise NSE Stock Price Prediction & Quantitative Portfolio Platform
# ==============================================================================

set -e

echo "🐳 ===================================================================="
echo "🚀 Deploying NSE Stock Prediction Platform to Cloud (AWS / DigitalOcean)"
echo "===================================================================="

# 1. Update Linux system packages
echo "📦 Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y

# 2. Install Docker & Docker Compose if missing
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    echo "🐙 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 3. Pull / Build containers and launch stack
echo "🏗️ Building and launching Docker Compose stack..."
sudo docker-compose down --remove-orphans 2>/dev/null || true
sudo docker-compose build --no-cache
sudo docker-compose up -d

echo "===================================================================="
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "🌐 Your Platform is now live 24/7 on your Cloud Server at:"
echo "👉 http://$(curl -s ifconfig.me):80"
echo "👉 http://$(curl -s ifconfig.me):5000"
echo "===================================================================="

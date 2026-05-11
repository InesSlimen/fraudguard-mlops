#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Checking health..."
curl -s "$BASE_URL/health"
echo

echo "Checking readiness..."
curl -s "$BASE_URL/ready"
echo

echo "Sending prediction..."
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 129.99,
    "ProductCD": "W",
    "card1": 12345,
    "card2": 321,
    "card3": 150,
    "card4": "visa",
    "card5": 226,
    "card6": "credit",
    "addr1": 204,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "DeviceType": "desktop",
    "DeviceInfo": "Windows"
  }'
echo
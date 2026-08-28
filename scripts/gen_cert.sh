#!/usr/bin/env bash
# 生成自签 TLS 证书（演示/内网环境用）。
# 用法：bash scripts/gen_cert.sh [域名/IP ...]    # 可选，把访问域名加进 SAN（默认仅 localhost）
#   例：内网 IP 访问  bash scripts/gen_cert.sh 192.168.1.10
#   例：自定义域名  bash scripts/gen_cert.sh my.cs.local
# 公网生产请改用 Let's Encrypt / 企业 CA 的受信任证书（certbot 自动续期），
# 并保持路径一致（certs/server.crt + certs/server.key）即可免改 nginx 配置。
set -euo pipefail
cd "$(dirname "$0")/.."

# Git Bash 会把 /CN=... 当路径转换（→ D:/software/Git/CN=...），禁用 MSYS 路径转换
export MSYS_NO_PATHCONV=1

mkdir -p certs

# 组装 SAN：默认 localhost；额外参数（域名/IP）逐个追加（现代浏览器对无 SAN 的证书更严格）
san="DNS:localhost,IP:127.0.0.1"
cn="localhost"
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9.]+$ ]]; then
    san="$san,IP:$arg"
  else
    san="$san,DNS:$arg"
    cn="$arg"
  fi
done

openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -subj "/CN=$cn" \
  -addext "subjectAltName=$san"

echo "证书已生成（有效期 1 年）: certs/server.crt + certs/server.key"
echo "SAN: $san"
echo "HTTPS 入口: https://localhost:8443（自签证书浏览器需手动信任；curl 用 -k）"

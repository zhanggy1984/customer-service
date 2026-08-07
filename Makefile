# 智能客服系统 — Makefile
# Windows 无 make 时可直接使用等价 docker compose 命令。
.PHONY: up down ps logs build backend-frontend test

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f

build:
	docker compose build

# 构建前端静态产物（挂载到 nginx）
frontend:
	cd frontend && npm run build

test:
	cd backend && pytest tests/ -v

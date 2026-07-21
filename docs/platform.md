# Backing Platform — Nextcloud

`nextcloud-agent` is a **client** of a Nextcloud instance. This page provides a
Docker recipe for deploying one locally to serve as the target of `NEXTCLOUD_URL`.
For production topologies, follow the upstream
[Nextcloud documentation](https://docs.nextcloud.com/).

!!! note "Backing-system recipe"
    Each connector in the ecosystem follows the same convention — a
    `docs/platform.md` recipe for the system it integrates with, accompanied by a
    sample Compose stack that mirrors
    [`services/`](https://github.com/Knuckles-Team). Systems offered only as a
    managed service have no local recipe.

## Single-node deployment (Compose)

Nextcloud publishes the official `nextcloud:apache` image. The following stack runs
the application server on `:8080` backed by MariaDB and Redis, mirroring the ecosystem
[`services/nextcloud/compose.yml`](https://github.com/Knuckles-Team):

```yaml
# docker/nextcloud.compose.yml
services:
  nextcloud:
    image: nextcloud:apache
    container_name: nextcloud
    hostname: nextcloud
    restart: unless-stopped
    ports:
      - "8080:80"
    environment:
      - MYSQL_HOST=db
      - MYSQL_DATABASE=nextcloud
      - MYSQL_USER=nextcloud
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - NEXTCLOUD_ADMIN_USER=${NEXTCLOUD_ADMIN_USER}
      - NEXTCLOUD_ADMIN_PASSWORD=${NEXTCLOUD_ADMIN_PASSWORD}
      - REDIS_HOST=redis
    volumes:
      - nextcloud_html:/var/www/html
      - nextcloud_data:/var/www/html/data
    depends_on:
      - db
      - redis

  db:
    image: mariadb:noble
    restart: unless-stopped
    environment:
      - MYSQL_DATABASE=nextcloud
      - MYSQL_USER=nextcloud
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
    volumes:
      - nextcloud_db:/var/lib/mysql

  redis:
    image: redis:alpine
    restart: unless-stopped
    volumes:
      - nextcloud_redis:/data

volumes:
  nextcloud_html:
  nextcloud_data:
  nextcloud_db:
  nextcloud_redis:
```

```bash
docker compose -f docker/nextcloud.compose.yml up -d

# Wait for the application to answer
curl -sf http://localhost:8080/status.php
```

## Connect nextcloud-agent

Create an app password for the connector (Settings → Security → Devices & sessions),
then point the environment at the instance:

```bash
export NEXTCLOUD_URL=http://localhost:8080
export NEXTCLOUD_USERNAME=your-user
export NEXTCLOUD_PASSWORD=your-app-password
# Select a named AgentConfig TLS profile or a runtime-only profile reference
# when the instance uses private PKI. Certificate verification cannot be disabled.

nextcloud-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

## Combined deployment

A combined stack places Nextcloud and the MCP server on one Docker network, so the
server reaches Nextcloud by container name:

```yaml
# docker/stack.compose.yml
services:
  nextcloud:
    image: nextcloud:apache
    hostname: nextcloud
    ports: ["8080:80"]
    environment:
      - MYSQL_HOST=db
      - REDIS_HOST=redis
    volumes: ["nextcloud_html:/var/www/html"]
    depends_on: [db, redis]

  db:
    image: mariadb:noble
    environment:
      - MYSQL_DATABASE=nextcloud
      - MYSQL_USER=nextcloud
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
    volumes: ["nextcloud_db:/var/lib/mysql"]

  redis:
    image: redis:alpine

  nextcloud-agent-mcp:
    image: knucklessg1/nextcloud-agent:mcp
    depends_on: [nextcloud]
    environment:
      - NEXTCLOUD_URL=http://nextcloud:80
      - NEXTCLOUD_USERNAME=${NEXTCLOUD_USERNAME}
      - NEXTCLOUD_PASSWORD=${NEXTCLOUD_PASSWORD}
      - TRANSPORT=streamable-http
      - HOST=0.0.0.0
      - PORT=8000
    ports: ["8000:8000"]

volumes:
  nextcloud_html:
  nextcloud_db:
```

```bash
docker compose -f docker/stack.compose.yml up -d
```

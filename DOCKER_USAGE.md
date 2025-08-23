# Docker 使用说明

Docker相关文件已整理到 `docker/` 目录下：

- `docker/Dockerfile` - Docker镜像定义文件
- `docker/Dockerfile.cn` - 使用国内镜像源的Dockerfile
- `docker/docker_build.sh` - 构建脚本
- `docker/docker_build_cn.sh` - 使用国内镜像源的构建脚本
- `docker/docker_run.sh` - 运行脚本
- `docker/docker-compose.yml` - Docker Compose配置文件
- `docker/docker-compose.cn.yml` - 使用国内镜像源的Docker Compose配置
- `docker/env.example` - 环境变量配置示例
- `docker/DOCKER_README.md` - 详细使用文档

## 快速开始

### 方式一：使用构建脚本

1. **构建镜像**：
   ```bash
   ./docker/docker_build.sh
   # 或使用国内镜像源
   ./docker/docker_build_cn.sh
   ```

2. **运行容器**：
   ```bash
   ./docker/docker_run.sh
   ```

### 方式二：使用Docker Compose（推荐）

1. **配置环境变量**：
   ```bash
   cp docker/env.example docker/.env
   # 编辑 docker/.env 设置你的路径
   ```

2. **构建并运行**：
   ```bash
   # 构建镜像
   docker-compose -f docker/docker-compose.yml build
   
   # 运行容器
   docker-compose -f docker/docker-compose.yml run --rm musubi-tuner
   ```

详细使用说明请查看 `docker/DOCKER_README.md`。

注意：`.dockerignore` 文件保留在项目根目录，这是Docker的要求。
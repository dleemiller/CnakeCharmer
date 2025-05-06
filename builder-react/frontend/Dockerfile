# frontend/Dockerfile
FROM node:18-alpine

# 1. Set workdir & copy manifest
WORKDIR /app
COPY package*.json ./

# 2. Install deps
RUN npm install

# 3. Copy source
COPY . .

# 4. Enable polling (for hot-reload in Docker on mac/linux)
ENV CHOKIDAR_USEPOLLING=true

# 5. Expose and start
EXPOSE 3000
CMD ["npm", "start"]

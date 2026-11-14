FROM node:26-slim

WORKDIR /app

# Non-root, using the `node` user node:26-slim already ships -- this
# container only serves the dev server and talks to the api over the
# network, it never needs root. --chown so npm install's own writes (and
# the anonymous node_modules/.next volumes docker-compose.yml mounts here,
# which Docker seeds from the image's existing ownership on first create)
# land owned by `node`, not root.
RUN chown node:node /app
USER node

COPY --chown=node:node package.json package-lock.json* ./
RUN npm install

COPY --chown=node:node . .

# .next doesn't exist in the image at all until `next dev` creates it --
# unlike node_modules (already node-owned above from `npm install`), so the
# anonymous .next volume docker-compose.yml mounts here has nothing to seed
# its ownership from and defaults to root, which `next dev` then can't
# write into as `node` (confirmed directly: EACCES on /app/.next/dev
# without this). Pre-creating it now, while still running as `node`, fixes
# that.
RUN mkdir -p .next

EXPOSE 3000

CMD ["npm", "run", "dev"]

# Ranil Mukesh MJ — Personal Portfolio

> **Co-Founder & CTO** at [PhobosQ](https://phobosq.com) & [k22.ai](https://k22.ai) — Architecting the infrastructure for autonomous enterprise intelligence.

A minimal, fast personal portfolio built with **Astro 5**, featuring a blog, dark/light theme toggle, and view transitions.

---

## 🚀 Live

| Deployment | URL |
|---|---|
| Primary (GitHub Pages) | [ranilmukesh.github.io](https://ranilmukesh.github.io) |
| Mirror | [astro-ranilmukesh-website2026](https://github.com/ranilmukesh/astro-ranilmukesh-website2026) |

---

## 📁 Project Structure

```
/
├── public/                  # Static assets (favicon, images)
├── src/
│   ├── components/
│   │   ├── Header.astro     # Navigation + dark/light theme toggle
│   │   └── Footer.astro     # Site footer
│   ├── content/
│   │   ├── config.ts        # Content collections schema
│   │   └── blog/            # Markdown blog posts
│   │       ├── Manual Data Science Delusion.md
│   │       ├── deterministic-ai.md
│   │       ├── rust-agent coding agent's mind.md
│   │       └── rust-agent.md
│   ├── layouts/
│   │   └── Layout.astro     # Base HTML shell (meta, theme script, ViewTransitions)
│   ├── pages/
│   │   ├── index.astro      # Homepage (Hero, Projects, Research, Expertise)
│   │   ├── expertise.astro  # Full technical stack page
│   │   ├── contact.astro    # Contact page
│   │   └── blog/            # Blog listing & post routes
│   ├── styles/
│   │   └── global.css       # CSS custom properties, base styles
│   └── env.d.ts             # TypeScript env declarations
├── astro.config.mjs         # Astro configuration
├── package.json
└── tsconfig.json
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Framework | [Astro 5](https://astro.build) |
| Styling | Vanilla CSS (custom properties, no framework) |
| Blog | Astro Content Collections (Markdown) |
| Transitions | Astro View Transitions API |
| Deployment | GitHub Pages |
| Language | TypeScript |

---

## 📄 Pages

### `/` — Homepage
- **Hero** — Name, title, social links (Email, LinkedIn, GitHub, X, LeetCode)
- **Philosophy** — Engineering approach & signature quote
- **Featured Work** — 10 projects (SIRUS, Yazhi, QEO, Luna, TOM, RAGFlow-AI, AutoML Studio, NeoAI, Startup Validator, Distributed RAG Engine)
- **Research** — 3 academic projects (License Plate Recognition, Quantum-SVM, Generative RL)
- **Open Source** — Contributions (NeoAI, RAGFlow-AI, Agno-AGI)
- **Technical Expertise** — Architecture, Infrastructure, Orchestration, Foundations
- **What I'm Building** — Voice Agents, Interview Platform, Yazhi, PhobosQ AutoSEO

### `/expertise` — Full Technical Stack
Detailed breakdown of all tools, frameworks, and languages.

### `/contact` — Contact
Reach-out form and direct contact links.

### `/blog` — Blog
Long-form technical writing on AI, agentic systems, and software engineering.

---

## 🖥 Development

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:4321)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🌐 Deployment & Remotes

This repo is pushed to **two** GitHub remotes simultaneously:

```bash
# Check remotes
git remote -v

# Push to both repos
git push origin main   # → astro-ranilmukesh-website2026
git push pages main    # → ranilmukesh.github.io
```

| Remote | Repository |
|---|---|
| `origin` | `github.com/ranilmukesh/astro-ranilmukesh-website2026` |
| `pages` | `github.com/ranilmukesh/ranilmukesh.github.io` |

---

## ✨ Features

- ⚡ **Zero JS by default** — Astro ships no runtime JS unless needed
- 🌙 **Theme toggle** — Dark/light mode persisted in `localStorage`, flash-free via inline script
- 🔄 **View Transitions** — Smooth page-to-page animations via Astro's built-in API
- 📱 **Responsive** — Mobile-first layout with CSS Grid and media queries
- 🔍 **SEO-ready** — Meta descriptions, semantic HTML, hidden keyword hints for crawlers
- 📝 **Content Collections** — Type-safe Markdown blog with schema validation

---

## 📬 Contact

- **Email**: [ranilmukesh117@gmail.com](mailto:ranilmukesh117@gmail.com)
- **LinkedIn**: [linkedin.com/in/ranilmukesh](https://linkedin.com/in/ranilmukesh)
- **GitHub**: [github.com/ranilmukesh](https://github.com/ranilmukesh)
- **X**: [x.com/ranilmukesh](https://x.com/ranilmukesh)

---

<p align="center">Built with Astro · Deployed on GitHub Pages</p>

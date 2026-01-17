# Jeffrey Wei - Personal Website

Personal website built with Jekyll and the Minimal Mistakes theme. Features projects, technical blog, and information about my research in AI and physics.

## Local Development

1. Install dependencies:
```bash
bundle install
```

2. Run local server:
```bash
bundle exec jekyll serve
```

3. Visit `http://localhost:4000/jeffrey-wei/`

## Adding Blog Posts

Create a new file in `_posts/technical/` or `_posts/personal/` with the format:

```
YYYY-MM-DD-title.md
```

Include frontmatter:

```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD
categories:
  - category-name
tags:
  - tag1
  - tag2
excerpt: "Brief description"
---

Your content here...
```

## Project Structure

- `_config.yml` - Site configuration
- `_data/navigation.yml` - Navigation menu
- `_pages/` - Main pages (about, projects, blog, cool-things)
- `_projects/` - Individual project descriptions
- `_posts/` - Blog posts (technical and personal)
- `assets/` - Images, CSS, and files

## Math Support

Use KaTeX syntax for math equations:

- Inline: `$x^2$`
- Display: `$$x^2$$`

## Deployment

The site automatically deploys to GitHub Pages when pushing to the `main` branch.

Visit: https://jwei302.github.io/jeffrey-wei/

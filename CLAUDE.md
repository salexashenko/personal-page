# CLAUDE.md - Guidelines for Personal Website

## Project Structure
- Static HTML/CSS website with some vanilla JavaScript
- Deployed via GitHub Pages (indicated by CNAME file)

## Development Commands
- No build process required (static site)
- Local preview: `python -m http.server` or `npx serve`
- Validate HTML: `npx html-validate "**/*.html"`

## Code Style Guidelines

### HTML
- Use 4 spaces for indentation
- Clean, minimalist structure
- Include proper meta tags and charset
- Page structure: #menu for navigation, #content for content area

### CSS
- Centralized in static/style.css
- Custom font: Helvetica Now Display
- Color scheme: Primary blue #002fa7 for links
- Responsive design with 600px fixed-width content area
- Animation: Underline hover effect for links

### JavaScript
- Vanilla JS only (no frameworks)
- Descriptive variable names (camelCase)
- Document code with appropriate comments for complex logic
- Keep scripts at bottom of HTML before closing body tag

### Best Practices
- Maintain consistent HTML structure across pages
- Test on multiple browsers and devices
- Keep image sizes optimized
- Preserve the minimal aesthetic throughout site changes
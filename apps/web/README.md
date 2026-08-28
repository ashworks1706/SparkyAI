# Sparky Campus Assistant

Sparky Campus Assistant is a Next.js-based web application designed to provide a seamless user experience with a modern UI powered by Tailwind CSS. This project is configured with TypeScript, ESLint, and PostCSS for a robust and maintainable development workflow.

## Features

- **Next.js Framework**: Built with the latest features of Next.js.
- **TypeScript Support**: Ensures type safety and better developer experience.
- **Tailwind CSS**: Fully styled with Tailwind CSS for rapid UI development.
- **ESLint Configuration**: Pre-configured with ESLint for consistent code quality.
- **PostCSS**: Integrated for advanced CSS processing.
- **ShadCN UI**: Configured with ShadCN for reusable components.
- **Icons**: Includes a variety of SVG icons in the `public` folder.

## Project Structure

```
.
├── app/
│   ├── favicon.ico
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── public/
│   ├── favicon.ico
│   ├── file.svg
│   ├── globe.svg
│   ├── next.svg
│   ├── og-image.png
│   ├── placeholder.svg
│   ├── vercel.svg
│   └── window.svg
├── src/
│   ├── App.css
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   ├── vite-env.d.ts
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   └── pages/
├── components.json
├── eslint.config.js
├── index.html
├── next.config.ts
├── package.json
├── postcss.config.js
├── tailwind.config.ts
├── tsconfig.app.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sparky-campus-assistant.git
   cd sparky-campus-assistant
   ```
2. Install dependencies:

   ```bash
   npm install
   ```

### Development

To start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

### Build

To build the project for production:

```bash
npm run build
```

### Linting

To run ESLint:

```bash
npm run lint
```

## Configuration

### Tailwind CSS

The Tailwind CSS configuration is located in [`tailwind.config.ts`](tailwind.config.ts). The base styles are defined in [`src/index.css`](src/index.css).

### ESLint

The ESLint configuration is defined in [`eslint.config.js`](eslint.config.js). It includes support for React, TypeScript, and React Hooks.

### Icons

SVG icons are located in the [`public`](public/) folder and can be used in the application via the `Image` component from Next.js.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Next.js](https://nextjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [ShadCN UI](https://ui.shadcn.com/)
- [Vercel](https://vercel.com/)

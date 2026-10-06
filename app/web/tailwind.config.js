// Tailwind scans the templates for class names and builds only those into
// app/web/static/app.css (see the assets stage in the Dockerfile).
module.exports = {
  content: { relative: true, files: ["./templates/**/*.html"] },
};

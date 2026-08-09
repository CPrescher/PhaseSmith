const runtime = document.querySelector("#runtime");
const format = document.querySelector("#format");

try {
  const info = await window.__TAURI__.core.invoke("runtime_info");
  runtime.textContent = "Native desktop adapter connected";
  format.textContent = String(info.project_format_version);
} catch (error) {
  runtime.textContent = `Runtime connection failed: ${String(error)}`;
}

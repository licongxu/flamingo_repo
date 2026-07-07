/* Shared helpers: authenticated fetch, formatting, header nav highlight. */

const API = "/api/v1";

async function apiGet(path, params) {
  const url = new URL(API + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
    }
  }
  const headers = {};
  const token = localStorage.getItem("flamingo_token");
  if (token) headers["X-API-Key"] = token;
  const resp = await fetch(url, { headers });
  if (resp.status === 401) {
    const entered = prompt("This portal requires an API token (X-API-Key):");
    if (entered) {
      localStorage.setItem("flamingo_token", entered);
      return apiGet(path, params);
    }
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (e) { /* keep statusText */ }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

function fmtBytes(b) {
  if (b >= 1e12) return (b / 1e12).toFixed(2) + " TB";
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  return (b / 1e3).toFixed(0) + " kB";
}

function fmtNum(x) {
  if (x === null || x === undefined) return "";
  if (typeof x !== "number") return String(x);
  if (Number.isInteger(x)) return x.toLocaleString();
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(4);
  return x.toPrecision(6);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}

document.addEventListener("DOMContentLoaded", () => {
  const here = window.location.pathname.replace(/\/$/, "") || "/index.html";
  document.querySelectorAll("header nav a").forEach((a) => {
    const target = a.getAttribute("href").replace(/\/$/, "");
    if (here.endsWith(target) || (target === "index.html" && here === "/index.html")) {
      a.classList.add("active");
    }
  });
});

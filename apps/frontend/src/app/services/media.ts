// src/app/services/media.ts
import type { AnomalyCase } from "../data/mockData";
import { API_BASE } from "../api/http";

export type ImageVariant = "original" | "heatmap" | "overlay";

function normalizeUrl(path?: string | null): string | null {
  if (!path) return null;
  const p = String(path).trim();
  if (!p) return null;

  // 이미 절대 URL이면 그대로 사용
  if (p.startsWith("http://") || p.startsWith("https://")) return p;

  // "/..." 형태면 API_BASE 붙임
  if (p.startsWith("/")) return `${API_BASE}${p}`;

  // 그 외(로컬 파일 경로 등)는 프론트에서 안전하게 처리 불가 → null
  return null;
}

export function getCaseImageUrl(c: AnomalyCase, variant: ImageVariant): string | null {
  if (variant === "original") return normalizeUrl(c.image_path);
  if (variant === "heatmap") return normalizeUrl(c.heatmap_path);
  return normalizeUrl(c.overlay_path);
}

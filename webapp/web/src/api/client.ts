import axios from "axios";
import type {
  AlertItem,
  AuditItem,
  Camera,
  PersonDetail,
  PersonSummary,
  SearchResponse,
  Stats,
  User,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
export const TOKEN_KEY = "surv_token";

export const mediaUrl = (p?: string | null) => (p ? `${API_BASE}${p}` : "");
export const streamUrl = (camId: number, overlay = false) =>
  `${API_BASE}/cameras/${camId}/stream?token=${encodeURIComponent(
    localStorage.getItem(TOKEN_KEY) || ""
  )}${overlay ? "&overlay=1" : ""}`;

export const api = axios.create({ baseURL: API_BASE });

api.interceptors.request.use((cfg) => {
  const t = localStorage.getItem(TOKEN_KEY);
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.status === 401 && !location.pathname.startsWith("/login")) {
      localStorage.removeItem(TOKEN_KEY);
      location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ── Auth ──────────────────────────────────────────────────────────────────
export async function login(email: string, password: string) {
  const body = new URLSearchParams({ username: email, password });
  const { data } = await axios.post(`${API_BASE}/auth/login`, body, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  return data as { access_token: string; token_type: string; user: User };
}
export const getMe = () => api.get<User>("/auth/me").then((r) => r.data);

// ── Persons ─────────────────────────────────────────────────────────────────
export interface PersonListParams {
  status?: string;
  camera?: number;
  gender?: string;
  age_range?: string;
  ethnicity?: string;
  glasses?: string;
  has_description?: boolean;
  q?: string;
  limit?: number;
  offset?: number;
}
export const listPersons = (params: PersonListParams = {}) =>
  api
    .get("/persons", { params })
    .then((r) => r.data as { total: number; limit: number; offset: number; items: PersonSummary[] });

export const getPerson = (id: string) =>
  api.get<PersonDetail>(`/persons/${id}`).then((r) => r.data);

// ── Search ──────────────────────────────────────────────────────────────────
export const searchText = (query: string, top_k = 10) =>
  api.post<SearchResponse>("/search/text", { query, top_k }).then((r) => r.data);

export const searchFilters = (body: Record<string, unknown>) =>
  api.post<SearchResponse>("/search/filters", body).then((r) => r.data);

export const searchImage = (file: File, mode: "body" | "face", top_k = 5) => {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", mode);
  fd.append("top_k", String(top_k));
  return api.post<SearchResponse>("/search/image", fd).then((r) => r.data);
};

// ── Dashboard ───────────────────────────────────────────────────────────────
export const getStats = () => api.get<Stats>("/stats").then((r) => r.data);
export const getAlerts = () =>
  api.get<{ total: number; items: AlertItem[] }>("/alerts").then((r) => r.data);
export const getCameras = () =>
  api
    .get<{ cameras: Camera[]; overlap_groups: number[][]; overlay_available: boolean }>("/cameras")
    .then((r) => r.data);
export const getPendingMerges = () =>
  api.get<{ items: any[] }>("/merges/pending").then((r) => r.data);
export const getAudit = () =>
  api.get<{ items: AuditItem[] }>("/corrections/audit").then((r) => r.data);

// ── Audio intelligence ───────────────────────────────────────────────────────
export interface AudioStatus {
  enabled: boolean;
  available: boolean;
  missing_core: string[];
  missing_speaker: string[];
  whisper_model?: string;
  target_lang?: string;
  error?: string;
}
export interface AudioResult {
  transcript: string;
  speaker: string;
  speaker_known: boolean;
  language: {
    code?: string | null;
    name?: string | null;
    confidence?: number | null;
    whisper_lang?: string | null;
    whisper_conf?: number | null;
  };
  translation: null | { text: string; engine?: string; target_lang?: string; source_lang?: string };
  threat: { detected: boolean; keywords: string[]; alert_sent: boolean };
  whisper: { model?: string | null; confidence?: number | null; segments?: number | null };
  source?: string;
}

export const getAudioStatus = () => api.get<AudioStatus>("/audio/status").then((r) => r.data);
export const getAudioSamples = () =>
  api.get<{ items: { name: string; size: number }[] }>("/audio/samples").then((r) => r.data);
export const audioSampleUrl = (name: string) =>
  `${API_BASE}/audio/samples/${encodeURIComponent(name)}`;
export const analyzeAudioSample = (sample: string, language = "auto") => {
  const fd = new FormData();
  fd.append("sample", sample);
  fd.append("language", language);
  return api.post<AudioResult>("/audio/analyze", fd).then((r) => r.data);
};
export const analyzeAudioFile = (file: File, language = "auto") => {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("language", language);
  return api.post<AudioResult>("/audio/analyze", fd).then((r) => r.data);
};

// ── Corrections ─────────────────────────────────────────────────────────────
export const editAttributes = (id: string, body: Record<string, string>) =>
  api.patch(`/persons/${id}/attributes`, body).then((r) => r.data);
export const redescribe = (id: string) =>
  api.post(`/persons/${id}/redescribe`).then((r) => r.data);
export const mergePersons = (keep_id: string, remove_id: string) =>
  api.post("/corrections/merge", { keep_id, remove_id }).then((r) => r.data);
export const splitPerson = (id: string, embedding_ids: number[], history_ids: number[] = []) =>
  api.post(`/persons/${id}/split`, { embedding_ids, history_ids }).then((r) => r.data);
export const deletePerson = (id: string) =>
  api.delete(`/persons/${id}`).then((r) => r.data);

// ── Chat / Assistant ─────────────────────────────────────────────────────────
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}
export interface ChatReply {
  session_id: string;
  reply: string;
  results?: import("./types").PersonSummary[] | null;
  open_person_id?: string | null;
  proposed_action?: { type: string; args: Record<string, string>; summary: string } | null;
}
export const postChat = (message: string, session_id?: string) =>
  api.post<ChatReply>("/chat", { message, session_id }).then((r) => r.data);

export const getChatMessages = (session_id: string) =>
  api
    .get<{ messages: ChatMessage[] }>(`/chat/sessions/${session_id}/messages`)
    .then((r) => r.data);

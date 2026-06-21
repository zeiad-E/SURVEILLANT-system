import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Bar, Eyebrow, GlassCard, Section, SectionHeading, Tag } from "./ui";

type Status = "live" | "planned";

interface Mod {
  title: string;
  status: Status;
  body: string;
  tech: string[];
  metrics?: { k: string; v: string }[];
}

const MODULES: Mod[] = [
  {
    title: "Person Re-Identification",
    status: "live",
    body: "Matches the same person across non-overlapping cameras using pose-aware OSNet appearance embeddings, with context-aware thresholds for same-camera, cross-camera and overlapping views.",
    tech: ["OSNet x1.0", "Market-1501", "FAISS"],
    metrics: [{ k: "Embedding", v: "512-d" }, { k: "Index", v: "sub-ms" }],
  },
  {
    title: "Face Detection & Recognition",
    status: "live",
    body: "InsightFace detects faces and produces 512-d face embeddings for a named watchlist and 'returning person' badges — kept in an isolated store so it never alters body identity.",
    tech: ["InsightFace", "buffalo_l", "ONNX Runtime"],
  },
  {
    title: "Demographic Analysis",
    status: "live",
    body: "Estimates age range and gender from faces, plus a ResNet-18 ethnicity classifier — surfaced as filterable attributes across the gallery.",
    tech: ["InsightFace", "ResNet-18"],
    metrics: [{ k: "Gender", v: "99%" }, { k: "Age MAE", v: "~3.0" }, { k: "Ethnicity", v: "77%" }],
  },
  {
    title: "Emotion Detection",
    status: "planned",
    body: "Facial-expression analysis to flag distress or agitation as an auxiliary threat signal feeding the alert pipeline.",
    tech: ["CNN", "FER"],
  },
  {
    title: "Violence Detection",
    status: "live",
    body: "A CNN-LSTM watches rolling 16-frame clips for aggressive motion, runs in its own daemon thread, and raises alerts with saved snapshots, clips and optional email.",
    tech: ["ResNet-50 + BiLSTM", "16-frame clips"],
    metrics: [{ k: "Window", v: "16 frames" }],
  },
  {
    title: "Appearance Attributes",
    status: "live",
    body: "A local vision-language model writes a rich body description (clothing, colors, accessories) — emitting only what is actually visible, never hallucinated attributes.",
    tech: ["Qwen2.5-VL", "Ollama", "CPU"],
    metrics: [{ k: "Backend", v: "Local" }],
  },
  {
    title: "Natural-Language Search",
    status: "live",
    body: "Body descriptions are embedded with a sentence model; a free-text query is embedded and cosine-ranked, so you search by meaning rather than keywords.",
    tech: ["all-MiniLM-L6-v2", "cosine"],
  },
  {
    title: "Speech & Translation",
    status: "live",
    body: "Whisper transcribes speech, a multi-engine voter detects the language, and MarianMT (with a Google fallback) translates it to any target language — all running locally.",
    tech: ["Whisper", "MarianMT", "langdetect"],
    metrics: [{ k: "Languages", v: "30+" }],
  },
  {
    title: "Speaker Recognition",
    status: "live",
    body: "ECAPA-TDNN voiceprints identify who is speaking via a nearest-neighbour embedding bank with adaptive per-speaker thresholds and test-time augmentation.",
    tech: ["ECAPA-TDNN", "SpeechBrain"],
  },
  {
    title: "Threat Detection",
    status: "live",
    body: "Fuses visual violence scores with spoken-threat keyword detection across ~30 languages into a single prioritized alert feed with severity, source and media.",
    tech: ["Fusion", "Multilingual", "Alert log"],
  },
];

function StatusPill({ status }: { status: Status }) {
  return status === "live" ? (
    <span className="site-mono inline-flex items-center gap-1.5 rounded-md border border-emerald-400/40 bg-emerald-400/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-emerald-300">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 site-pulse" /> Live
    </span>
  ) : (
    <span className="site-mono inline-flex items-center gap-1.5 rounded-md border border-slate-500/40 bg-white/[0.03] px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
      Planned
    </span>
  );
}

export default function Modules() {
  const live = MODULES.filter((m) => m.status === "live").length;
  return (
    <div className="relative">
      {/* Background image merged into the top of the page */}
      <img src="/modules-core.png" alt="" className="site-merged-img site-merged-img-top h-[600px]" />
      
      <Section className="relative z-10 pt-16 sm:pt-20">
        <SectionHeading
          eyebrow="// System modules"
          title="Ten modules. One unified pipeline."
          intro="Each module is independent and flag-gated — the system degrades gracefully when a model is absent, and never lets one signal corrupt another."
        />
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag>{live} live</Tag>
          <Tag>{MODULES.length - live} planned</Tag>
        </div>
      </Section>

      <Section className="mt-12 space-y-4">
        {MODULES.map((m, i) => (
          <GlassCard key={m.title} hover className="p-6 sm:p-7">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
              <div className="site-index text-5xl sm:text-6xl">{String(i + 1).padStart(2, "0")}</div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="text-xl font-semibold text-white">{m.title}</h3>
                  <StatusPill status={m.status} />
                </div>
                <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">{m.body}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {m.tech.map((t) => (
                    <span
                      key={t}
                      className="site-mono rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 text-[11px] text-cyan-200/80"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
              {m.metrics && (
                <div className="flex shrink-0 gap-6 sm:flex-col sm:gap-3 sm:border-l sm:border-white/10 sm:pl-6">
                  {m.metrics.map((mt) => (
                    <div key={mt.k}>
                      <div className="text-lg font-bold text-emerald-300 site-glow-text">{mt.v}</div>
                      <div className="site-mono text-[10px] uppercase tracking-wider text-slate-500">
                        {mt.k}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </GlassCard>
        ))}
      </Section>

      {/* Performance at a glance */}
      <Section className="mt-20">
        <GlassCard className="overflow-hidden">
          <div className="border-b border-white/10 px-6 py-4">
            <Eyebrow>// Module performance at a glance</Eyebrow>
          </div>
          <div className="divide-y divide-white/5">
            {[
              { n: "Person Re-ID", v: 92 },
              { n: "Face recognition", v: 95 },
              { n: "Gender", v: 99 },
              { n: "Ethnicity", v: 77 },
              { n: "Violence detection", v: 88 },
              { n: "NL search relevance", v: 90 },
            ].map((r) => (
              <div key={r.n} className="flex items-center gap-4 px-6 py-3">
                <span className="w-44 shrink-0 text-sm text-slate-300">{r.n}</span>
                <div className="flex-1">
                  <Bar value={r.v} />
                </div>
                <span className="site-mono w-12 text-right text-sm text-emerald-300">{r.v}%</span>
              </div>
            ))}
          </div>
        </GlassCard>
        <p className="site-mono mt-4 text-center text-xs text-slate-500">
          Indicative figures from internal evaluation on the WiseNet set.
        </p>
        <div className="mt-10 text-center">
          <Link to="/architecture" className="site-btn site-btn-ghost">
            See how they connect <ArrowRight size={16} />
          </Link>
        </div>
      </Section>
    </div>
  );
}

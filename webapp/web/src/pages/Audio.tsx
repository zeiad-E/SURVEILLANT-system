import { type ReactNode, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  AudioLines,
  Languages,
  Mic,
  ShieldAlert,
  ShieldCheck,
  Upload,
  UserCheck,
} from "lucide-react";
import {
  analyzeAudioFile,
  analyzeAudioSample,
  audioSampleUrl,
  getAudioSamples,
  getAudioStatus,
  type AudioResult,
} from "../api/client";
import { Card, Empty, PageHeader, Spinner } from "../components/ui";

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-ink-800 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-emerald-200/40">{label}</div>
      <div className="mt-0.5 text-sm text-emerald-100">{children}</div>
    </div>
  );
}

function ResultPanel({ r }: { r: AudioResult }) {
  const threat = r.threat?.detected;
  return (
    <div className="space-y-4">
      {/* threat banner */}
      <div
        className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-sm ${
          threat
            ? "border-red-500/40 bg-red-500/10 text-red-200"
            : "border-emerald/30 bg-emerald/10 text-emerald-glow"
        }`}
      >
        {threat ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
        {threat ? (
          <span>
            Spoken threat detected — keywords:{" "}
            <span className="font-mono">{r.threat.keywords.join(", ")}</span>
            {r.threat.alert_sent && " · alert emailed"}
          </span>
        ) : (
          <span>No spoken threats detected.</span>
        )}
      </div>

      {/* transcript */}
      <div>
        <div className="mb-1 text-xs uppercase tracking-wider text-emerald-200/40">Transcript</div>
        <p className="rounded-lg border border-line bg-ink-900/60 px-4 py-3 text-sm leading-relaxed text-emerald-100">
          {r.transcript || <span className="text-emerald-200/30">(no speech recognized)</span>}
        </p>
      </div>

      {/* meta grid */}
      <div className="grid grid-cols-2 gap-2">
        <Field label="Language">
          <span className="inline-flex items-center gap-1.5">
            <Languages size={13} className="text-cyan-300" />
            {r.language?.name || r.language?.code || "—"}
            {r.language?.confidence != null && (
              <span className="text-emerald-200/40">
                {" "}
                {(r.language.confidence * 100).toFixed(0)}%
              </span>
            )}
          </span>
        </Field>
        <Field label="Speaker">
          <span className="inline-flex items-center gap-1.5">
            <UserCheck size={13} className={r.speaker_known ? "text-emerald-glow" : "text-emerald-200/40"} />
            {r.speaker || "—"}
          </span>
        </Field>
        <Field label={`Translation${r.translation?.target_lang ? ` → ${r.translation.target_lang}` : ""}`}>
          {r.translation?.text ? (
            <>
              {r.translation.text}
              {r.translation.engine && (
                <span className="ml-1 text-[10px] text-emerald-200/30">[{r.translation.engine}]</span>
              )}
            </>
          ) : (
            <span className="text-emerald-200/30">—</span>
          )}
        </Field>
        <Field label="Whisper">
          {r.whisper?.model || "—"}
          {r.whisper?.confidence != null && (
            <span className="text-emerald-200/40"> · {(r.whisper.confidence * 100).toFixed(0)}%</span>
          )}
        </Field>
      </div>
    </div>
  );
}

const LANGS: { code: string; label: string }[] = [
  { code: "auto", label: "Auto-detect" },
  { code: "en", label: "English" },
  { code: "ar", label: "Arabic" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "es", label: "Spanish" },
  { code: "it", label: "Italian" },
  { code: "ru", label: "Russian" },
  { code: "tr", label: "Turkish" },
  { code: "zh", label: "Chinese" },
];

export default function Audio() {
  const status = useQuery({ queryKey: ["audio-status"], queryFn: getAudioStatus });
  const samples = useQuery({ queryKey: ["audio-samples"], queryFn: getAudioSamples });
  const [selected, setSelected] = useState<string | null>(null);
  const [lang, setLang] = useState("auto");

  const mSample = useMutation({ mutationFn: (name: string) => analyzeAudioSample(name, lang) });
  const mFile = useMutation({ mutationFn: (file: File) => analyzeAudioFile(file, lang) });
  const active = mSample.isPending || mFile.isPending;
  const result = (mFile.data || mSample.data) as AudioResult | undefined;
  const error = (mFile.error || mSample.error) as any;

  const available = status.data?.available;
  const missing = [
    ...(status.data?.missing_core ?? []),
  ];

  return (
    <>
      <PageHeader
        title="Audio Intelligence"
        subtitle="Transcribe · detect language · translate · identify speaker · flag spoken threats"
        right={
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-emerald-200/50">
              <Languages size={14} className="text-cyan-300" />
              <select
                className="input w-auto py-1 text-xs"
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                title="Force the spoken language (Auto-detect can be wrong for accented/bilingual speech)"
              >
                {LANGS.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </select>
            </label>
            <span className={`chip ${available ? "border-emerald/40 text-emerald-glow" : "text-amber-300 border-amber-400/40"}`}>
              {available ? "online" : "offline"}
            </span>
          </div>
        }
      />

      {!available && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-xs text-amber-200/80">
          <AudioLines size={15} className="mt-0.5 flex-none text-amber-300" />
          <span>
            Audio analysis is {status.data?.enabled ? "enabled but missing packages" : "disabled"}.
            {missing.length > 0 && (
              <>
                {" "}
                Install: <code className="rounded bg-ink-900 px-1">pip install {missing.join(" ")} torchaudio speechbrain transformers langdetect deep-translator</code>
              </>
            )}
            {!status.data?.enabled && (
              <> Set <code className="rounded bg-ink-900 px-1">ENABLE_AUDIO_ANALYSIS = True</code> in settings. </>
            )}
            You can still queue a clip — the server will report what's missing.
          </span>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {/* ── Input ── */}
        <Card className="p-4">
          <div className="mb-3 text-xs uppercase tracking-wider text-emerald-200/50">Demo samples</div>
          {samples.isLoading ? (
            <Spinner className="text-emerald" />
          ) : (samples.data?.items.length ?? 0) === 0 ? (
            <Empty>No bundled samples found.</Empty>
          ) : (
            <div className="space-y-2">
              {samples.data!.items.map((s) => (
                <div
                  key={s.name}
                  className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${
                    selected === s.name ? "border-emerald/50 bg-emerald/5" : "border-line"
                  }`}
                >
                  <button
                    onClick={() => setSelected(s.name)}
                    className="flex-1 text-left font-mono text-xs text-emerald-100"
                  >
                    {s.name}
                  </button>
                  <audio controls preload="none" src={audioSampleUrl(s.name)} className="h-8" />
                  <button
                    className="btn-primary px-2 py-1 text-xs"
                    disabled={active}
                    onClick={() => {
                      setSelected(s.name);
                      mFile.reset();
                      mSample.mutate(s.name);
                    }}
                  >
                    <Mic size={13} /> Analyze
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="mt-4 border-t border-line pt-4">
            <div className="mb-2 text-xs uppercase tracking-wider text-emerald-200/50">Or upload a clip</div>
            <label className="btn-ghost cursor-pointer">
              <Upload size={15} />
              Choose audio file
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm"
                className="hidden"
                disabled={active}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    setSelected(null);
                    mSample.reset();
                    mFile.mutate(f);
                  }
                }}
              />
            </label>
            <p className="mt-2 text-xs text-emerald-200/40">
              wav / mp3 / m4a / ogg / flac / webm. Transcription can take a while on CPU.
            </p>
          </div>
        </Card>

        {/* ── Result ── */}
        <Card className="p-4">
          <div className="mb-3 text-xs uppercase tracking-wider text-emerald-200/50">Analysis</div>
          {active ? (
            <div className="grid h-48 place-items-center gap-3 text-center">
              <Spinner className="text-emerald" />
              <div className="text-xs text-emerald-200/50">
                Running Whisper + speaker ID… this can take a moment on CPU.
              </div>
            </div>
          ) : error ? (
            <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-3 text-sm text-red-300">
              {typeof error?.response?.data?.detail === "object"
                ? error.response.data.detail.message + " " + (error.response.data.detail.hint ?? "")
                : String(error?.response?.data?.detail ?? "Analysis failed.")}
            </div>
          ) : result ? (
            <ResultPanel r={result} />
          ) : (
            <Empty>Pick a sample or upload a clip to analyze.</Empty>
          )}
        </Card>
      </div>
    </>
  );
}

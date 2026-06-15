import { useState, useCallback } from "react";
import type { DiagnosisResult } from "../lib/types";
import { api } from "../lib/api";

export type DiagnosisMode = "fusion" | "image" | "audio";

export function useDiagnosis() {
  const [mode, setMode] = useState<DiagnosisMode>("fusion");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imageUrl = imageFile ? URL.createObjectURL(imageFile) : null;
  const audioUrl = audioFile ? URL.createObjectURL(audioFile) : null;

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("mode", mode);
      if (imageFile) fd.append("image_file", imageFile);
      if (audioFile) fd.append("audio_file", audioFile);

      const data = await api.diagnosis.run(fd);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Diagnosis error");
    } finally {
      setLoading(false);
    }
  }, [mode, imageFile, audioFile]);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setImageFile(null);
    setAudioFile(null);
  }, []);

  const canSubmit =
    !loading &&
    ((mode === "fusion" && imageFile && audioFile) ||
      (mode === "image" && imageFile) ||
      (mode === "audio" && audioFile));

  return {
    mode,
    setMode,
    imageFile,
    setImageFile,
    audioFile,
    setAudioFile,
    imageUrl,
    audioUrl,
    result,
    loading,
    error,
    run,
    reset,
    canSubmit,
  };
}

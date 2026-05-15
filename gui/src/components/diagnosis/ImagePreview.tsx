interface Props {
  url: string | null;
}

export default function ImagePreview({ url }: Props) {
  if (!url) return null;

  return (
    <div className="rounded-xl border border-gray-800 overflow-hidden bg-black">
      <img
        src={url}
        alt="X-quang preview"
        className="w-full max-h-72 object-contain"
      />
    </div>
  );
}

import { Server, Network } from "lucide-react";

interface Props {
  connectedToServer: boolean;
  connectedToFlower: boolean;
}

export default function ConnectionStatus({ connectedToServer, connectedToFlower }: Props) {
  const items = [
    { label: "Server", ok: connectedToServer, icon: Server },
    { label: "Flower", ok: connectedToFlower, icon: Network },
  ];

  return (
    <div className="flex gap-4">
      {items.map(({ label, ok, icon: Icon }) => (
        <div
          key={label}
          className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
            ok ? "bg-green-600/10 text-green-400" : "bg-red-600/10 text-red-400"
          }`}
        >
          <span className={`flex h-2 w-2 rounded-full ${ok ? "bg-green-400" : "bg-red-400"}`} />
          <Icon size={14} />
          {label}
        </div>
      ))}
    </div>
  );
}

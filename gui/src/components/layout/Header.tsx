import { Activity } from "lucide-react";

export default function Header() {
  const clientName = import.meta.env.VITE_CLIENT_NAME || "Client";
  const clientId = import.meta.env.VITE_CLIENT_ID || "1";
  return (
    <header className="flex items-center justify-between border-b border-gray-800 bg-gray-900/50 px-6 py-3">
      <div className="flex items-center gap-2">
        <Activity size={16} className="text-green-400" />
        <span className="text-sm text-gray-300">
          Client: <span className="font-medium text-white">{clientName}_{clientId}</span>
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="flex h-2 w-2 rounded-full bg-green-400" />
        <span className="text-xs text-gray-500">Đã kết nối</span>
      </div>
    </header>
  );
}

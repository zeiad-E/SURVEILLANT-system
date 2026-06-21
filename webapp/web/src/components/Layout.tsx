import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  AudioLines,
  LayoutDashboard,
  LogOut,
  MessageCircle,
  Search,
  ShieldAlert,
  Users,
  Video,
} from "lucide-react";
import { useAuth } from "../auth/AuthContext";

const NAV = [
  { to: "/app", label: "Overview", icon: LayoutDashboard },
  { to: "/app/cameras", label: "Live Cams", icon: Video },
  { to: "/app/people", label: "People", icon: Users },
  { to: "/app/search", label: "Search", icon: Search },
  { to: "/app/audio", label: "Audio", icon: AudioLines },
  { to: "/app/assistant", label: "Assistant", icon: MessageCircle },
  { to: "/app/alerts", label: "Alerts", icon: ShieldAlert },
];

export default function Layout() {
  const { user, logout } = useAuth();
  return (
    <div className="flex h-full">
      <aside className="flex w-60 flex-col border-r border-line bg-ink-800/70 p-4">
        <div className="mb-8 flex items-center gap-2 px-2">
          <Activity className="text-emerald-glow" />
          <span className="text-lg font-bold tracking-widest text-emerald-glow">SURVEILLANT</span>
        </div>
        <nav className="flex flex-1 flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/app"}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? "bg-emerald/15 text-emerald-glow shadow-glow"
                    : "text-emerald-100/70 hover:bg-ink-600"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-4 border-t border-line pt-4">
          <div className="px-2 text-xs text-emerald-200/60">{user?.email}</div>
          <div className="px-2 text-[10px] uppercase tracking-wider text-emerald-200/30">
            {user?.role}
          </div>
          <button onClick={logout} className="btn-ghost mt-2 w-full">
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

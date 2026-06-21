import { type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import Login from "./pages/Login";
import Overview from "./pages/Overview";
import People from "./pages/People";
import SearchPage from "./pages/Search";
import LiveCams from "./pages/LiveCams";
import Alerts from "./pages/Alerts";
import Assistant from "./pages/Assistant";
import Audio from "./pages/Audio";
// Public marketing / academic-showcase site
import SiteLayout from "./site/SiteLayout";
import Home from "./site/Home";
import Modules from "./site/Modules";
import Architecture from "./site/Architecture";
import Ethics from "./site/Ethics";
import Team from "./site/Team";

function Protected({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading)
    return (
      <div className="grid h-full place-items-center">
        <Spinner className="text-emerald" />
      </div>
    );
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      {/* Public site — no login required */}
      <Route element={<SiteLayout />}>
        <Route path="/" element={<Home />} />
        <Route path="/modules" element={<Modules />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/ethics" element={<Ethics />} />
        <Route path="/team" element={<Team />} />
      </Route>

      <Route path="/login" element={<Login />} />

      {/* Authenticated console */}
      <Route
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route path="/app" element={<Overview />} />
        <Route path="/app/cameras" element={<LiveCams />} />
        <Route path="/app/people" element={<People />} />
        <Route path="/app/search" element={<SearchPage />} />
        <Route path="/app/audio" element={<Audio />} />
        <Route path="/app/assistant" element={<Assistant />} />
        <Route path="/app/alerts" element={<Alerts />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

import "@xyflow/react/dist/style.css";
import { Suspense, useEffect } from "react";
import { RouterProvider } from "react-router-dom";
import { LoadingPage } from "./pages/LoadingPage";
import router from "./routes";
import { useDarkStore } from "./stores/darkStore";
import { LANGFLOW_ACCESS_TOKEN } from "./constants/constants";
import { setLocalStorage } from "./utils/local-storage-util";

function useLTTokenFromUrl() {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const ltToken = params.get("lt_token");
    if (ltToken) {
      setLocalStorage(LANGFLOW_ACCESS_TOKEN, ltToken);
      params.delete("lt_token");
      const newUrl = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
      window.history.replaceState({}, "", newUrl);
    }
  }, []);
}

export default function App() {
  const dark = useDarkStore((state) => state.dark);
  useEffect(() => {
    if (!dark) {
      document.getElementById("body")!.classList.remove("dark");
    } else {
      document.getElementById("body")!.classList.add("dark");
    }
  }, [dark]);

  useLTTokenFromUrl();

  return (
    <Suspense fallback={<LoadingPage />}>
      <RouterProvider router={router} />
    </Suspense>
  );
}

import { useEffect, useState } from "react";
import { isAxiosError } from "axios";
import predictionService from "../services/predictionService";

export default function useApi() {
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    predictionService.health()
      .then(() => {
        if (active) {
          setOnline(true);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setOnline(false);
          setError(isAxiosError(err) ? err.message : "API offline");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const checkHealth = async () => {
    setLoading(true);
    try {
      await predictionService.health();
      setOnline(true);
      setError(null);
    } catch (err) {
      setOnline(false);
      setError(isAxiosError(err) ? err.message : "API offline");
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    online,
    error,
    refresh: checkHealth,
  };
}
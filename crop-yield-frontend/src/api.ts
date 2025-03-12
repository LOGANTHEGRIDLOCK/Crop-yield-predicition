interface CropPredictionData {
  region: string;
  soil_type: string;
  crop: string;
  avg_temp: number;
  avg_rainfall: number;
  weather: string;
  days_to_harvest: number;
  fertilizer_used: boolean;
  irrigation_used: boolean;
  }
  
export const predictYield = async (formData: CropPredictionData) => {
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error predicting yield:", error);
      return { error: "Failed to fetch prediction" };
    }
  };
  
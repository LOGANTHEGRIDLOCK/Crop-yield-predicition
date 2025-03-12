interface PredictionProps {
    prediction: number | null;
  }
  const PredictionResult = ({ prediction }: PredictionProps) => {
    if (prediction === null) return null;
    return (
      <div className="mt-4 p-2 bg-green-100 border border-green-400">
        <h3>Estimated Yield: {prediction.toFixed(2)} tons per hectare</h3>
      </div>
    );
  };
  export default PredictionResult;
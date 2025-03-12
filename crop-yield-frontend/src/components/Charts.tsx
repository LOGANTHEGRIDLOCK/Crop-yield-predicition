import { useState, useEffect } from "react";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  LineChart, Line, CartesianGrid, Legend, 
  ComposedChart, Area
} from "recharts";

// Define interfaces for props and historical data
interface PredictionHistory {
  date: string;
  yield: number;
  cropType: string;
  region: string;
}

interface ChartsProps {
  predictedYield: number;
  cropType: string;
  region: string;
  averageYield: number;  // Fetched from backend
  optimalYield: number;   // Fetched from backend
}

const Charts: React.FC<ChartsProps> = ({ 
  predictedYield, 
  cropType, 
  region, 
  averageYield, 
  optimalYield 
}) => {
  // State for time period selection
  const [timePeriod, setTimePeriod] = useState<'week' | 'month' | 'year'>('month');
  
  // State for prediction history
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistory[]>([]);

  // Fetch prediction history from the backend instead of using localStorage
  useEffect(() => {
    fetch("http://localhost:8000/history")
      .then(response => response.json())
      .then(data => setPredictionHistory(data.history))
      .catch(error => console.error("Error fetching history:", error));
  }, []);

  // Generate comparison data
  const comparisonData = [
    { name: "Your Prediction", yield: predictedYield },
    { name: `Avg. ${cropType}`, yield: averageYield },
    { name: "Optimal Yield", yield: optimalYield }
  ];

  // Process historical data based on selected time period
  const getFilteredHistory = () => {
    const now = new Date();
    return predictionHistory.filter(item => {
      const predDate = new Date(item.date);
      if (timePeriod === 'week') return (now.getTime() - predDate.getTime()) <= 7 * 24 * 60 * 60 * 1000;
      if (timePeriod === 'month') return (now.getTime() - predDate.getTime()) <= 30 * 24 * 60 * 60 * 1000;
      return (now.getTime() - predDate.getTime()) <= 365 * 24 * 60 * 60 * 1000;
    }).map(item => ({
      ...item,
      date: new Date(item.date).toLocaleDateString()
    })).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  };

  const historyData = getFilteredHistory();

  const archiveHistory = async () => {
    try {
      await fetch("http://localhost:8000/archive", { method: "POST" });
      setPredictionHistory([]);  // Clear current history view
    } catch (error) {
      console.error("Error archiving history:", error);
    }
  };

  const generateDaysToHarvestData = () => {
    const data = [];
    const totalDays = 100; // Assuming 100 days to harvest
    const interval = 5; // Update every 5 days
  
    for (let day = 0; day <= totalDays; day += interval) {
      let growthFactor;
      
      if (day < totalDays * 0.2) {
        // Slow initial growth phase
        growthFactor = day / (totalDays * 0.4);
      } else if (day < totalDays * 0.8) {
        // Rapid middle growth phase
        growthFactor = 0.5 + (day - totalDays * 0.2) / (totalDays * 1.2);
      } else {
        // Slower final growth phase
        growthFactor = 0.9 + (day - totalDays * 0.8) / (totalDays * 2);
      }
  
      if (growthFactor > 1) growthFactor = 1; // Max growth factor is 1 (100%)
  
      data.push({
        day,
        projectedYield: parseFloat((predictedYield * growthFactor).toFixed(2)),
      });
    }
    
    return data;
  };

  const daysToHarvestData = generateDaysToHarvestData();


  return (
    <div className="bg-slate-950">
    <div className="w-full max-w-6xl mx-auto mt-6">
      {/* Time period selector */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex rounded-md shadow-sm" role="group">
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium border ${
              timePeriod === 'week' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'
            } rounded-l-lg`}
            onClick={() => setTimePeriod('week')}
          >
            Week
          </button>
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium border-t border-b border-r ${
              timePeriod === 'month' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'
            }`}
            onClick={() => setTimePeriod('month')}
          >
            Month
          </button>
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium border-t border-b border-r ${
              timePeriod === 'year' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'
            } rounded-r-lg`}
            onClick={() => setTimePeriod('year')}
          >
            Year
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Comparison Chart */}
         <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-center mb-2">Yield Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value) => [`${value} tons/ha`, 'Yield']} />
              <Legend />
              <Bar dataKey="yield" name="Yield (tons/ha)" fill="#4CAF50" barSize={50} />
            </BarChart>
          </ResponsiveContainer>
        </div>

          {/* Historical Yield Trend Chart */}
        <div className="bg-white p-4 rounded-lg shadow mt-6">
          <h3 className="text-lg font-semibold text-center mb-2">Historical Yield Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historyData}>
              <XAxis dataKey="date" />
              <YAxis />
              <CartesianGrid strokeDasharray="3 3" />
              <Tooltip formatter={(value) => [`${value} tons/ha`, 'Yield']} />
              <Legend />
              <Line type="monotone" dataKey="yield" name="Yield (tons/ha)" stroke="#007bff" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Days to Harvest Projection Chart */}
      <div className="bg-white p-4 rounded-lg shadow mt-6">
        <h3 className="text-lg font-semibold text-center mb-2">Growth Projection by Days to Harvest</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={daysToHarvestData}>
            <XAxis dataKey="day" label={{ value: 'Days', position: 'insideBottomRight', offset: -5 }} />
            <YAxis label={{ value: 'Yield (tons/ha)', angle: -90, position: 'insideLeft' }} />
            <CartesianGrid strokeDasharray="3 3" />
            <Tooltip formatter={(value) => [`${value} tons/ha`, 'Projected Yield']} />
            <Area 
              type="monotone" 
              dataKey="projectedYield" 
              name="Projected Yield"
              fill="#82ca9d" 
              stroke="#4CAF50" 
              fillOpacity={0.3} 
            />
            <Line 
              type="monotone" 
              dataKey="projectedYield" 
              name="Projected Yield"
              stroke="#4CAF50" 
              strokeWidth={2} 
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {predictionHistory.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Prediction History Summary</h3>
            <button 
              onClick={archiveHistory}
              className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
            >
              Clear History
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead>
                <tr>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Crop Type
                  </th>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Region
                  </th>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Yield (tons/ha)
                  </th>
                </tr>
              </thead>
              <tbody>
                {predictionHistory.slice(-5).reverse().map((item, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                   <td className="py-2 px-4 text-sm text-gray-500">
                      {new Date(item.date).toLocaleDateString()}
                    </td>
                    <td className="py-2 px-4 text-sm text-gray-500">{item.cropType}</td>
                    <td className="py-2 px-4 text-sm text-gray-500">{item.region}</td>
                    <td className="py-2 px-4 text-sm text-gray-500">{item.yield.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      <div className="bg-white p-4 rounded-lg shadow mt-6">
        <h3 className="text-lg font-semibold text-center mb-4">Yield Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-green-50 p-4 rounded-lg border border-green-100">
            <h4 className="text-sm font-medium text-green-800">Current Prediction</h4>
            <p className="text-2xl font-bold text-green-600">{predictedYield.toFixed(2)} tons/ha</p>
            <p className="text-xs text-green-500 mt-1">For {cropType} in {region}</p>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
            <h4 className="text-sm font-medium text-blue-800">Average Yield</h4>
            <p className="text-2xl font-bold text-blue-600">{averageYield.toFixed(2)} tons/ha</p>
            <p className="text-xs text-blue-500 mt-1">Regional average for {cropType}</p>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
            <h4 className="text-sm font-medium text-purple-800">Potential Improvement</h4>
            <p className="text-2xl font-bold text-purple-600">
              {((predictedYield / averageYield - 1) * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-purple-500 mt-1">
              Compared to regional average
            </p>
          </div>
        </div>
      </div>

      {/* Recommendations based on data */}
      <div className="bg-white p-4 rounded-lg shadow mt-6">
        <h3 className="text-lg font-semibold mb-4">Yield Improvement Recommendations</h3>
        <div className="space-y-4">
          {predictedYield < averageYield ? (
            <>
              <div className="flex items-start">
                <div className="flex-shrink-0 h-5 w-5 text-red-500">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h4 className="text-sm font-medium text-gray-900">Below Average Yield</h4>
                  <p className="mt-1 text-sm text-gray-600">
                    Your predicted yield is {((averageYield - predictedYield) / averageYield * 100).toFixed(1)}% below the regional average for {cropType}.
                  </p>
                </div>
              </div>
              <div className="pl-8 space-y-2">
                <p className="text-sm text-gray-800">Consider the following improvements:</p>
                <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                  <li>Optimize soil fertility with targeted fertilizer application</li>
                  <li>Review irrigation practices and schedules</li>
                  <li>Consider crop rotation or companion planting</li>
                  <li>Implement pest and disease management strategies</li>
                </ul>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-start">
                <div className="flex-shrink-0 h-5 w-5 text-green-500">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h4 className="text-sm font-medium text-gray-900">Above Average Yield</h4>
                  <p className="mt-1 text-sm text-gray-600">
                    Your predicted yield is {((predictedYield - averageYield) / averageYield * 100).toFixed(1)}% above the regional average for {cropType}.
                  </p>
                </div>
              </div>
              <div className="pl-8 space-y-2">
                <p className="text-sm text-gray-800">To maintain or improve your yield:</p>
                <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                  <li>Continue with current best practices</li>
                  <li>Consider soil testing to maintain optimal conditions</li>
                  <li>Monitor weather patterns for timely interventions</li>
                  <li>Document your methods for future reference</li>
                </ul>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Footer with disclaimer */}
      <div className="mt-8 text-center text-xs text-gray-500">
        <p>Yield predictions are based on historical data and environmental factors.</p>
        <p>Actual yields may vary based on weather conditions, management practices, and other factors.</p>
      </div>
    </div>
    </div>
  );
};

export default Charts;
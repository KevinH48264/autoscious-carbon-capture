import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';
import { Sidebar } from './Sidebar/Sidebar';

const App = () => {
  const [loading, setLoading] = useState(true);
  const [papersData, setPapersData] = useState([]);
  const [edgesData, setEdgesData] = useState([]);
  const [clusterData, setClusterData] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [isPlotReady, setIsPlotReady] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(process.env.REACT_APP_PAPERS_JSON_URL)
          .then(response => response.ok ? response.json() : [])
          .catch(error => {
              console.log('Fetch papers failed:', error, process.env.REACT_APP_PAPERS_JSON_URL);
              return [];
          }),

      fetch(process.env.REACT_APP_EDGES_JSON_URL)
          .then(response => response.ok ? response.json() : [])
          .catch(error => {
              console.log('Fetch edges failed:', error, process.env.REACT_APP_EDGES_JSON_URL);
              return [];
          }),

      fetch(process.env.REACT_APP_TAXONOMY_JSON_URL)
          .then(response => response.ok ? response.json() : [])
          .catch(error => {
              console.log('Fetch taxonomy failed:', error, process.env.REACT_APP_TAXONOMY_JSON_URL);
              return [];
          })
    ])
    .then(([papers, edges, taxonomy]) => {
        setPapersData(papers);
        setEdgesData(edges);
        setClusterData(taxonomy);
        setLoading(false);  // set loading to false after fetching data
    });
  }, []);

  // Waiting screen for fetch
  if (loading) {
      return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          color: '#000000',
          fontSize: '2em', // Increase or decrease this to your liking
        }}>
          Loading in database of papers...
        </div>
      ); 
  }

  return (
    <div className="App" style={{ padding: "0px", margin: "0", overflow: "hidden", position: "relative" }}>
      {papersData.length > 0 && edgesData.length > 0 && clusterData.length > 0 && <ResearchPaperPlot papersData={papersData} edgesData={edgesData} clusterData={clusterData} setSelectedPaper={setSelectedPaper} isPlotReady={isPlotReady} setIsPlotReady={setIsPlotReady}/>}
      <Sidebar selectedPaper={selectedPaper}/>
    </div>
  );
};

export default App;
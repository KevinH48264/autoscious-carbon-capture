import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';

const App = () => {
  const [papersData, setPapersData] = useState([]);
  const [edgesData, setEdgesData] = useState([]);
  const [clusterData, setClusterData] = useState([]);

  useEffect(() => {
    fetch('2000/gpt_classified_papers_2000.json')
      .then(response => response.json())
      .then(json => {
        setPapersData(json);
          console.log("Papers data:", json);
      })

    fetch('2000/top_2_edges_2000.json')
      .then(response => response.json())
      .then(json => {
        setEdgesData(json);
          console.log("Edges data:", json);
      })

    fetch('2000/taxonomy_2000.json')
      .then(response => response.json())
      .then(json => {
        setClusterData(json);
          console.log("Tree data:", json);
      })    
  }, []);

  return (
    <div className="App" style={{ padding: "0px", margin: "0", overflow: "hidden" }}>
      {papersData.length > 0 && edgesData.length > 0 && clusterData.length > 0 && <ResearchPaperPlot papersData={papersData} edgesData={edgesData} clusterData={clusterData} />}
    </div>
  );
};

export default App;
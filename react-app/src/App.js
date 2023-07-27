import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';
import { Sidebar } from './Sidebar/Sidebar';

const App = () => {
  const [papersData, setPapersData] = useState([]);
  const [edgesData, setEdgesData] = useState([]);
  const [clusterData, setClusterData] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);

  useEffect(() => {
    fetch('latest/latest_papers.json')
      .then(response => response.json())
      .then(json => {
        setPapersData(json);
          console.log("Papers data:", json);
      })

    fetch('latest/latest_edges.json')
      .then(response => response.json())
      .then(json => {
        setEdgesData(json);
          console.log("Edges data:", json);
      })

    fetch('latest/latest_taxonomy.json')
      .then(response => response.json())
      .then(json => {
        setClusterData(json);
          console.log("Tree data:", json);
      })    
  }, []);

  return (
    <div className="App" style={{ padding: "0px", margin: "0", overflow: "hidden", position: "relative" }}>
      {papersData.length > 0 && edgesData.length > 0 && clusterData.length > 0 && <ResearchPaperPlot papersData={papersData} edgesData={edgesData} clusterData={clusterData} setSelectedPaper={setSelectedPaper}/>}
      <Sidebar selectedPaper={selectedPaper}/>
    </div>
  );
};

export default App;
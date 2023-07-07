import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';

const App = () => {
  const [papersData, setPapersData] = useState([]);
  const [edgesData, setEdgesData] = useState([]);
  const [clusterData, setClusterData] = useState([]);

  useEffect(() => {
    fetch('output_100_tsne.json')
      .then(response => response.json())
      .then(json => {
        setPapersData(json);
          console.log("Papers data:", json);
      })

    fetch('pruned_tree_w_main_topic.json')
      .then(response => response.json())
      .then(json => {
        setClusterData(json);
          console.log("Tree data:", json);
      })

    fetch('edges.json')
      .then(response => response.json())
      .then(json => {
        setEdgesData(json);
          console.log("Edges data:", json);
      })
  }, []);

  return (
    <div className="App" style={{ padding: "0px", margin: "0", overflow: "hidden" }}>
      {edgesData.length > 0 && <ResearchPaperPlot papersData={papersData} edgesData={edgesData} clusterData={clusterData} />}
    </div>
  );
};

export default App;
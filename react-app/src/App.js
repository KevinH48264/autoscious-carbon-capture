import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';

const App = () => {
  const [papersData, setPapersData] = useState([]);
  const [topicsData, setTopicsData] = useState([]);

  useEffect(() => {
    fetch('output_100_tsne.json')
      .then(response => response.json())
      .then(json => {
        setPapersData(json);
          console.log("Papers data:", json);
      })

    fetch('topic_100_tsne.json')
      .then(response => response.json())
      .then(json => {
        setTopicsData(json);
          console.log("Topics data:", json);
      })
  }, []);

  return (
    <div className="App" style={{ padding: "0px", margin: "0", overflow: "hidden" }}>
      {/* <div 
        id="tooltip" 
        style={{
          position: 'absolute', 
          display: 'none', 
          backgroundColor: 'white', 
          color: 'black', 
          padding: '10px', 
          borderRadius: '5px', 
          zIndex: 1000,
          pointerEvents: 'none',
          userSelect: 'none'
        }} 
      /> */}
      {topicsData.length > 0 && <ResearchPaperPlot papersData={papersData} topicsData={topicsData}/>}
    </div>
  );
};

export default App;
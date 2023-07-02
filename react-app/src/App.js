import React, { useState, useEffect } from 'react';
import ResearchPaperPlot from './PapersMap';

const App = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('output_100_tsne.json')
      .then(response => response.json())
      .then(json => {
          setData(json);
          console.log("Data:", json);
      })
  }, []);

  return (
    <div className="App">
      {data.length > 0 && <ResearchPaperPlot data={data} />}
    </div>
  );
};

export default App;
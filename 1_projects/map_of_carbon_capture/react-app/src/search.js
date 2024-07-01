import React, { useState } from 'react';

const SemanticSearchBar = ({ papersData }) => {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  //const [top10Results, setTop10Results] = useState([]);

  const handleInputChange = (event) => {
    const { value } = event.target;
    setQuery(value);
  };



    const handleSearch = () => {
    // api call
    console.log("Sending data:", { query });
    fetch('http://localhost:8000/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query:query })
    })
      .then((response) => response.json())
      .then((data) => {
        // top 10 results
        setSearchResults(data);
        console.log('Top 10 Search Results:', data);
      })
      .catch((error) => {
        console.error('Error performing search:', error);
      });
  };
     return (
    <div>
      <h1>Search Bar</h1>
      <div className="semantic-search-bar">
        <input
          type="text"
          placeholder="Search..."
          value={query}
          onChange={handleInputChange}
        />
        <button onClick={handleSearch}>Search</button>
      </div>
      {searchResults.length > 0 && (
        <div>
          <h2>Search Results</h2>
          <ul>
            {searchResults.map((result, index) => (
              <li key={index}>{result.id}, Title: {result.title}, Score: {result.score}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
    );
};

export default SemanticSearchBar;


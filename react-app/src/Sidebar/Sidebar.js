// Import necessary libraries
import React, { useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTimes, faBars } from '@fortawesome/free-solid-svg-icons';
import './Sidebar.css';

// Sidebar component
export const Sidebar = ({ selectedPaper }) => {
  const [sidebarActive, setSidebarActive] = useState(true);
  const [sidebarStyle, setSidebarStyle] = useState({ left: '0px' });

  // define open and close sidebar functions
  const openSidebar = () => {
    setSidebarStyle({ left: '0px' });
    setSidebarActive(true);
  }

  const closeSidebar = () => {
    setSidebarStyle({ left: '-300px' });
    setSidebarActive(false);
  }

  // effect to handle opening and closing sidebar
  useEffect(() => {
    if (selectedPaper) {
      openSidebar();
    } else {
      closeSidebar();
    }
  }, [selectedPaper]);

  return (
    <div style={{ position: "absolute", top: 0, left: 0, zIndex: 150, height: "100vh"}}>
      <button className="open-button" onClick={openSidebar} style={{ 
        position: "absolute",
        top: "25px",
        left: "10px",
        display: sidebarActive ? 'none' : 'block',
        backgroundColor: 'transparent', 
        border: 'none', 
        color: 'white',
        height: "40px",
        width: "40px",
        fontSize: "1.5rem",
        cursor: "pointer"
      }}>
        <FontAwesomeIcon icon={faBars} />
      </button>

      <aside id="sidebar" style={{
        position: "absolute",
        left: sidebarStyle.left,
        top: "0",
        height: "100%",
        width: "300px",
        backgroundColor: "#362E28",
        opacity: 0.95,
        boxShadow: "0px 0px 4px var(--color-d-shadow-10)",
        padding: "var(--scale-8-3) var(--scale-8-3) 0 var(--scale-8-3)",
        boxSizing: "border-box",
        overflow: "visible",
        color: "white",
        transition: "all 0.5s"
      }}>
        <div class="sidebarWrapper" style={{
          height: "100%",
          width: "100%",
          paddingRight: "4px",
        }}>
          <button class="close-button" onClick={closeSidebar} style={{ 
            position: "absolute",
            height: "40px", 
            width: "40px",
            top: "25px",
            left: "10px",
            fontSize: "1.5rem",
            backgroundColor: 'transparent', 
            border: 'none', 
            cursor: "pointer",
            color: 'white'
          }}>
            <FontAwesomeIcon icon={faTimes}/>
          </button>

          <div className="scrollableContent" style={{
            height: "85vh",
            overflowY: "auto",
            margin: '80px 10px 20px 20px',
            paddingRight: '10px',
            fontSize: '1rem'
          }}>
            <div style={{ marginBottom: '20px', fontWeight: 'bold' }}>Title: {selectedPaper && selectedPaper.title}</div>
            <div style={{ marginBottom: '20px' }}>Authors</div>
            <div style={{ marginBottom: '20px' }}>Year, Topic</div>
            <div style={{ marginBottom: '20px' }}># Citations: {selectedPaper && selectedPaper.citationCount}</div>
            <div style={{ marginBottom: '20px' }}>Abstract: {selectedPaper && selectedPaper.abstract}</div>
          </div>
        </div>

        
      </aside>
    </div>
  );
}

export default Sidebar;
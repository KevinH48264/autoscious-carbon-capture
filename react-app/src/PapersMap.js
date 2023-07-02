import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { EventSystem } from "@pixi/events";

const ResearchPaperPlot = ({ data }) => {
  const pixiContainer = useRef();

  useEffect(() => {
    const app = new PIXI.Application({ 
			width: window.innerWidth,
			height: window.innerHeight,
			resolution: window.devicePixelRatio || 1,
			autoDensity: true,
			backgroundColor: 0x121212,
			resizeTo: window,
    });

    if (pixiContainer.current && pixiContainer.current.childNodes.length > 0) {
      pixiContainer.current.replaceChild(app.view, pixiContainer.current.childNodes[0]);
    } else {
      pixiContainer.current.appendChild(app.view);
    }

    if (!("events" in app.renderer)) {
      app.renderer.addSystem(PIXI.EventSystem, "events");
    }
      
    const viewport = new Viewport({ 
      screenWidth: window.innerWidth,
			screenHeight: window.innerHeight,
      worldWidth: window.innerWidth,
			worldHeight: window.innerHeight,
      ticker: app.ticker,
      events: app.renderer.events
    });
    viewport.sortableChildren = true;
		viewport.moveCenter(0, 0);
    viewport.drag()
    .pinch()
    .wheel()
    // .clamp({ direction: "all" })
    // .clampZoom({ minScale: 0.5, maxScale: 1 })
    .decelerate();

    app.stage.addChild(viewport);

    // Calculate the min and max values just once
    const minX = Math.min(...data.map(p => p.x));
    const maxX = Math.max(...data.map(p => p.x));
    const minY = Math.min(...data.map(p => p.y));
    const maxY = Math.max(...data.map(p => p.y));

    // scale the data to fit within the worldWidth and worldHeight
    const scaleX = d => (d - minX) / (maxX - minX) * viewport.worldWidth;
    const scaleY = d => (d - minY) / (maxY - minY) * viewport.worldHeight;

    // Draw all circles in one go
    const circles = new PIXI.Graphics();
    data.forEach(paper => {
      circles.beginFill(0xFF0000);
      circles.drawCircle(scaleX(paper['x']), scaleY(paper['y']), 5);
      circles.endFill();
    });
    viewport.addChild(circles);

    // Create text labels and store them in an array for easier access
    const labels = data.map(paper => {
      const text = new PIXI.Text(paper['title'], { fontFamily : 'Arial', fontSize: 12, fill : 0xffffff, align : 'center' });
      text.position.set(scaleX(paper['x']), scaleY(paper['y']));
      text.visible = false;  // Initially hide all labels
      viewport.addChild(text);
      return text;
    });

    // Update label visibility based on zoom level
    viewport.on('zoomed', (e) => {
      const shouldShowLabels = e.viewport.scale.x > 0.5;  // Adjust the threshold as needed
      labels.forEach(label => label.visible = shouldShowLabels);
    });
  }, [data]);

  return <div ref={pixiContainer} />;
};

export default ResearchPaperPlot;
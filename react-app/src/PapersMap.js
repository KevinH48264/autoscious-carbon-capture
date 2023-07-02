import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';

const ResearchPaperPlot = ({ papersData, topicsData }) => {
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

    const viewport = new Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldWidth: window.innerWidth,
      worldHeight: window.innerHeight,
      ticker: app.ticker,
      events: app.renderer.events,
      stopPropagation: true,
    });
    viewport.sortableChildren = true;
    viewport.moveCenter(0, 0);
    viewport.drag().pinch().wheel().decelerate()
      // .clampZoom({ minWidth: 50, maxWidth: 5000})
      .setZoom(0.25)
      .moveCenter(0, 0);
    app.stage.addChild(viewport);

    // Calculate the min and max values just once
    const minX = Math.min(...papersData.map((paper) => paper.x), ...topicsData.map((topic) => topic.x));
    const maxX = Math.max(...papersData.map((paper) => paper.x), ...topicsData.map((topic) => topic.x));
    const minY = Math.min(...papersData.map((paper) => paper.y), ...topicsData.map((topic) => topic.y));
    const maxY = Math.max(...papersData.map((paper) => paper.y), ...topicsData.map((topic) => topic.y));

    // scale the data to fit within the worldWidth and worldHeight
    const scaleX = (d) => ((d - minX) / (maxX - minX)) * viewport.worldWidth;
    const scaleY = (d) => ((d - minY) / (maxY - minY)) * viewport.worldHeight;

    // console.log("topicsData", topicsData)
    let paperNodes = papersData.map(({title, x, y, citationCount}) => ({title, x, y, citationCount}))
    let topicNodes = topicsData.map(({topic, x, y, citationCount}) => ({title: topic, x, y, citationCount}))
    let nodes = paperNodes.concat(topicNodes);
    // console.log("nodes", nodes)

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, viewport) => {
      nodes.forEach((node) => {
        if(!node.circle) {
            node.circle = new PIXI.Graphics();
            node.circle.beginFill(0xff0000);
            node.circle.drawCircle(scaleX(node.x), scaleY(node.y), 5);
            node.circle.endFill();
            viewport.addChild(node.circle);
        } else {
            node.circle.visible = true; // make it visible if it already exists
        }

        if(!node.text) {
            node.text = new PIXI.Text(node.title, {
              fontFamily: 'Arial',
              fontSize: 12,
              fill: 0xffffff,
              align: 'center',
            });
            node.text.position.set(scaleX(node.x), scaleY(node.y));
            node.text.visible = false; // Initially hide all text
            viewport.addChild(node.text);
        } else {
            node.text.visible = true; // make it visible if it already exists
        }
      });
    }

    // Update visibility of circles and text based on the current field of view and zoom level
    const updateNodes = () => {
      if (!nodes) return;

      // reset all nodes and labels graphics
      for (const node of nodes) {
        if (node.circle) { node.circle.visible = false };
        if (node.text) { node.text.visible = false };
			}

      // get the current field of view
      const viewport_bounds = viewport.getVisibleBounds();
			viewport_bounds.pad(viewport_bounds.width * 0.2);
      let vis_nodes = nodes.filter((node) =>
				viewport_bounds.contains(scaleX(node.x), scaleY(node.y))
			)

      // Take the top 15 visible nodes
      vis_nodes.sort((a, b) => {
				return b.citationCount - a.citationCount;
			});
      vis_nodes = vis_nodes.slice(0, 15);

      // Update visibility of nodes and labels
      drawNodes(vis_nodes, viewport);
    };

    // Update nodes based on ticker
    app.ticker.add(updateNodes)

  }, [papersData, topicsData]);

  return <div ref={pixiContainer} />;
};

export default ResearchPaperPlot;

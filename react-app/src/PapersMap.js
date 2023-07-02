import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';

const ResearchPaperPlot = ({ data }) => {
  const pixiContainer = useRef();

  const updateNodes = (nodes, viewport) => {
    for (const node of nodes) {
      node.graphics.visible = false;
    }
    const viewport_bounds = viewport.getVisibleBounds();
    viewport_bounds.pad(viewport_bounds.width * 0.2);

    let vis_nodes = nodes.filter((node) =>
      viewport_bounds.contains(node.x, node.y)
    );

    vis_nodes.sort((a, b) => {
      return b.citationCount - a.citationCount;
    });
    vis_nodes = vis_nodes.slice(0, 20);

    for (const node of vis_nodes) {
      node.update();
      node.graphics.visible = true;
    }
    
    const invisible_nodes = _.difference(nodes, vis_nodes);
    for (const node of invisible_nodes) {
      node.label.visible = false;
    }

    drawLabels(vis_nodes, viewport);
    drawImages(nodes, viewport);
    // line_container.setEdges(edges);
  }

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
    });
    viewport.sortableChildren = true;
    viewport.moveCenter(0, 0);
    viewport.drag().pinch().wheel().decelerate()
      .clampZoom({ minWidth: 50, maxWidth: 5000})
      .setZoom(0.25)
      .moveCenter(0, 0);
    app.stage.addChild(viewport);

    // Calculate the min and max values just once
    const minX = Math.min(...data.map((paper) => paper.x));
    const maxX = Math.max(...data.map((paper) => paper.x));
    const minY = Math.min(...data.map((paper) => paper.y));
    const maxY = Math.max(...data.map((paper) => paper.y));

    // scale the data to fit within the worldWidth and worldHeight
    const scaleX = (d) => ((d - minX) / (maxX - minX)) * viewport.worldWidth;
    const scaleY = (d) => ((d - minY) / (maxY - minY)) * viewport.worldHeight;

    let circles = [];
    let texts = [];

    // Create and add all circles and text to the viewport
    data.forEach((paper) => {
      const circle = new PIXI.Graphics();
      circle.beginFill(0xff0000);
      circle.drawCircle(scaleX(paper.x), scaleY(paper.y), 5);
      circle.endFill();
      viewport.addChild(circle);
      circles.push(circle);

      const text = new PIXI.Text(paper.title, {
        fontFamily: 'Arial',
        fontSize: 12,
        fill: 0xffffff,
        align: 'center',
      });
      text.position.set(scaleX(paper.x), scaleY(paper.y));
      text.visible = false; // Initially hide all text
      viewport.addChild(text);
      texts.push(text);
    });

    // Update visibility of circles and text based on the current field of view and zoom level
    const updateVisibility = () => {
      const zoomLevel = viewport.scale.x;
      const visibleBounds = viewport.getVisibleBounds();

      circles.forEach((circle, index) => {
        const paper = data[index];
        const isVisible =
          circle.worldTransform.a / zoomLevel > 0.5 &&
          visibleBounds.contains(circle.x, circle.y) &&
          isTopCitation(paper, data, 20);
        circle.visible = isVisible;
      });

      texts.forEach((text, index) => {
        const paper = data[index];
        const isVisible =
          text.worldTransform.a / zoomLevel > 0.5 &&
          visibleBounds.contains(text.x, text.y) &&
          isTopCitation(paper, data, 20);
        text.visible = isVisible;
      });
    };

    // Check if the paper is among the top citations in the data
    const isTopCitation = (paper, data, count) => {
      const sortedData = [...data].sort(
        (a, b) => b.citationCount - a.citationCount
      );
      const index = sortedData.findIndex((p) => p.paperId === paper.paperId);
      return index >= 0 && index < count;
    };

    // Update visibility on zoom and move events
    viewport.on('zoomed', updateVisibility);
    viewport.on('moved', updateVisibility);

    // Initial visibility update
    updateVisibility();

    return () => {
      // Clean up event listeners
      viewport.off('zoomed', updateVisibility);
      viewport.off('moved', updateVisibility);
    };
  }, [data]);

  return <div ref={pixiContainer} />;
};

export default ResearchPaperPlot;

import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { randomDarkModeColor } from './util';

const ResearchPaperPlot = ({ papersData, topicsData }) => {
  const pixiContainer = useRef();

  useEffect(() => {
    const app = new PIXI.Application({
      width: window.innerWidth,
      height: window.innerHeight - 1000,
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

    // For now, make the world square and even for better circle parsing
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
    viewport.drag().pinch().wheel().decelerate()
      .clamp({direction: 'all'})
      .clampZoom({ maxHeight: viewport.worldHeight, maxWidth: viewport.worldWidth})
      .setZoom(0.5)
      .moveCenter(viewport.worldWidth / 2, viewport.worldHeight / 2);
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
    console.log(scaleX(maxX), scaleX(minX), scaleY(maxY), scaleY(minY))
    // console.log("nodes", nodes)

    // const tooltip = document.getElementById('tooltip');
  
    // // Track the mouse and update the tooltip
    // const onMouseMove = (event) => {
    //   tooltip.style.left = `${event.data.global.x}px`;
    //   tooltip.style.top = `${event.data.global.y}px`;

    //   for (let node of topicNodes) {
    //     if (node.region.containsPoint(event.data.global)) {
    //       tooltip.textContent = node.title;
    //       tooltip.style.display = 'block';
    //       return;
    //     }
    //   }

    //   tooltip.style.display = 'none';
    // }

    // app.stage.interactive = true;
    // app.stage.on('mousemove', onMouseMove);

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, viewport) => {
      // Voronoi diagram?
      const delaunay = Delaunay.from(topicNodes.map((node) => [scaleX(node.x), scaleY(node.y)]));
      const voronoi = delaunay.voronoi([0, 0, viewport.worldWidth, viewport.worldHeight]);

      topicNodes.forEach((node, i) => {
        if(!node.region) {
          const region = voronoi.cellPolygon(i);
          console.log(i, region)
          const polygon = new PIXI.Graphics();

          polygon.beginFill(PIXI.utils.string2hex(randomDarkModeColor()), 0.5);
          polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
          polygon.endFill();

          // polygon.interactive = true;
          // polygon.on('mouseover', (event) => {
          //   // TODO: Maybe this can just be displayed on the top left corner? Or render it faster, I'm not sure
          //   console.log("MOUSEOVER", node.title)
          //   const tooltip = document.getElementById('tooltip');
          //   tooltip.style.display = 'block';
          //   tooltip.style.left = `${event.global.x}px`;
          //   tooltip.style.top = `${event.global.y}px`;
          //   tooltip.textContent = node.title;
          // });
          // polygon.on('mouseout', () => {
          //   console.log("MOUSEOUT", node.title)
          //   const tooltip = document.getElementById('tooltip');
          //   tooltip.style.display = 'hidden';
          // });

          node.region = polygon;
          viewport.addChild(polygon);
        } else {
          node.region.visible = true; // make it visible if it already exists
        }
      });

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
      // Reset all region graphics
      for (const node of topicNodes) {
        if (node.region) node.region.visible = false;
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
      vis_nodes = vis_nodes.slice(0, 12);

      // Update visibility of nodes and labels
      drawNodes(vis_nodes, viewport);
    };

    // Update nodes based on ticker
    app.ticker.add(updateNodes)

  }, [papersData, topicsData]);

  return <div className="pixiContainer" style={{ padding: "0px", margin: "0", overflow: "hidden", display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;

import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { randomDarkModeColor, rectIntersectsRect, sortPoints, getLeafClusters, flattenClusters, diamondPolygonColorSequence, multilineText, labelBounds, getColorAtZoomLevel, traverseCluster, calculateClusterCentroids, getVoronoiNeighbors, circleColorDiamondSequence } from './util';
import { computeLayout } from './layout';

const ResearchPaperPlot = ({ papersData, edgesData, clusterData }) => {
  const pixiContainer = useRef();
  PIXI.BitmapFont.from("TitleFont", { fill: 0x000000 }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });
  PIXI.BitmapFont.from("TopicFont", { fill: 0x000000 }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });

  useEffect(() => {
    const logging = true;
    console.log("papersData", papersData, "edgesData", edgesData, "clusterData", clusterData)

    // Compute force-directed layout of PaperNodes
    let paperNodes = papersData.map(({title, x, y, citationCount, paperId}) => ({title, x: x, y: y, citationCount, paperId}))
    let leafClusters = getLeafClusters(clusterData);
    const layout = computeLayout(paperNodes, edgesData, leafClusters);
    paperNodes = layout.paperNodes;
    const centerNodes = layout.centerNodes;
    const normalizedRadius = layout.normalizedRadius; // this should be used to determine the zoom, currently 230
    const zoomScale = normalizedRadius / 150;

    const app = new PIXI.Application({
      width: window.innerWidth,
      height: window.innerHeight - 1000,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
      backgroundColor: 0xD6EFFF,
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
    viewport.drag().pinch().wheel().decelerate().clampZoom({ minWidth: 50, maxHeight: viewport.worldHeight / zoomScale, maxWidth: viewport.worldWidth / zoomScale}).setZoom(zoomScale).moveCenter(0, 0)
    // viewport.clamp({direction: 'all'})
    app.stage.addChild(viewport);

    // Creates a map from cluster_id to main_topic
    const clusterMap = new Map();
    clusterData.forEach(cluster => traverseCluster(cluster, clusterMap));

    const min_scale = Math.min(...paperNodes.map((node) => Math.sqrt(node.citationCount)));
    const max_scale = Math.max(...paperNodes.map((node) => Math.sqrt(node.citationCount)));

    // Create dummy 'center' nodes and links to them for each leafCluster
    let centroidNodes = []
    calculateClusterCentroids(leafClusters, paperNodes, centroidNodes)
    
    // Calculating voronoi from true centroids of leaf clusters
    const extendFactor = 100 // hardcoding for circle design
    const delaunay = Delaunay.from(centroidNodes.map((node) => [node.x, node.y]));
    const minX = Math.min(...paperNodes.map((paper) => paper.x));
    const maxX = Math.max(...paperNodes.map((paper) => paper.x));
    const minY = Math.min(...paperNodes.map((paper) => paper.y));
    const maxY = Math.max(...paperNodes.map((paper) => paper.y));
    const voronoi = delaunay.voronoi([minX - extendFactor, minY - extendFactor, maxX + extendFactor, maxY + extendFactor]);

    // Adding a circle mask
    let farthestDistance = 0;
    paperNodes.forEach(node => {
      let distance = Math.sqrt(Math.pow(node.x, 2) + Math.pow(node.y, 2));
      if (distance > farthestDistance) {
        farthestDistance = distance;
      }
    });
    let circleMask = new PIXI.Graphics();
    circleMask.beginFill(0x000000); // You can fill with any color
    circleMask.drawCircle(0, 0, farthestDistance + 10);
    circleMask.endFill();
    const polygonContainer = new PIXI.Container();
    viewport.addChild(polygonContainer);
    polygonContainer.mask = circleMask;
    viewport.addChild(circleMask);

    // Ensuring parentIds extend to the farthest zoom
    const zoomLayers = 20
    leafClusters.forEach((node) => {
        let lastParentId = node.cluster_id; // fallback to node's cluster_id
        for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
            if (node.parents[zoomLevel] !== undefined) {
                lastParentId = node.parents[zoomLevel];
            } else {
                node.parents[zoomLevel] = lastParentId;
            }
        }
    });

    // Creates a map from parentId to nodes with that parentId
    const clusterGroups = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
        leafClusters.forEach((node, i) => {
            const parentId = node.parents[zoomLevel];
            const key = [parentId, zoomLevel];
            if (!clusterGroups.has(key)) {
                clusterGroups.set(key, []);
            }
            // Get the indices of Voronoi neighbors for the current leaf cluster
            const neighborIndices = getVoronoiNeighbors(voronoi, i);
            // Check if any of the neighbors belong to the same parent cluster
            const neighborNodes = neighborIndices
                .map(index => leafClusters[index])
                .filter(neighbor => neighbor && neighbor.parents[zoomLevel] === parentId);
            // Add the current node and its same-parent neighbors to the group
            clusterGroups.get(key).push(node, ...neighborNodes);
        });
    }

    // Calculates centroid for each cluster
    const clusterCentroids = new Map();
    for (let [[clusterId, zoomLevel], nodes] of clusterGroups) {
        let sumX = 0, sumY = 0;
        for (let node of nodes) {
            sumX += node.centroid_x;
            sumY += node.centroid_y;
        }
        const key = [clusterId, zoomLevel].toString();
        clusterCentroids.set(key, {x: sumX / nodes.length, y: sumY / nodes.length, layer: zoomLevel});
    }

    // Hardcoding (zoomLayers) a parent cluster mapping for voronois
    let parentColorMap = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
      // Setting parent cluster colors by cluster_id
      leafClusters.forEach(node => {
        let parentId = node.parents[zoomLevel];
        if (parentId) {
          if (!parentColorMap.has(parentId)) {
              parentColorMap.set(parentId, diamondPolygonColorSequence[parentId % diamondPolygonColorSequence.length]);
          }
        }
      });
    }
    parentColorMap.set(0, 0xffffff); // hardcoding because 0 isn't covered for some reason

    // Sort paperNodes by citationCount to prioritize showing higher citationCount papers
    paperNodes.sort((a, b) => b.citationCount - a.citationCount);

    console.log("clusterCentroids", clusterCentroids)

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, vis_cluster_centroids, viewport) => {
      let zoomLevel = Math.max(-1, Math.round(((viewport.scaled / zoomScale) - 1) * 5))
      let originalZoomLevel = zoomLevel;

      // Font size
      const bounds = viewport.getVisibleBounds();
      let min_font_size = bounds.width < bounds.height
          ? bounds.width / (28.3)
          : bounds.height / (50);
      let max_font_size = min_font_size * 1.7;

      const addedTextBounds = new Set();

      // Preview Zoom: Adding cluster polygons on the preview layer to the viewport
      leafClusters.forEach((node, i) => {
        // If no parentId, then take the highest parent key
        let parentId = node.parents[zoomLevel + 1];
        if (!parentId) {
          parentId = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
        }
        let fillColor = parentColorMap.get(parentId);

        const region = voronoi.cellPolygon(i);
        const polygon = new PIXI.Graphics();
        polygon.zIndex = 50;

        polygon.beginFill(fillColor, 0.75);
        polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
        polygon.endFill();

        node.region = polygon;
        polygonContainer.addChild(polygon);
      });

      // change zoomLayers to maxZoomLayer
      // Current Zoom: Adding the cluster text to viewport, and any in the next 3 layers
      for (let i = 0; i < 3; ++i) {
        vis_cluster_centroids.forEach((centroid, key) => {
          let [clusterId] = key.split(',').map(Number); 
          if (centroid.layer === zoomLevel + i) {
            let topCategory = "Unknown";
            if(clusterMap.has(clusterId)){
                topCategory = clusterMap.get(clusterId);
            }
            topCategory = topCategory.slice(1, -1);

            // Create new text
            let current_centroid_font_size = max_font_size / (1.1 ** (zoomLevel - originalZoomLevel));
            let current_centroid_text = multilineText(topCategory, 15);

            // Check for font size bounds
            // if (current_centroid_font_size < min_font_size) {
            //   return
            // }

            // Not allowing more than 20 labels
            if (addedTextBounds.length > 20) {
              return
            }

            // Check for overlaps with existing labels
            let current_zoom_text_bound = labelBounds(current_centroid_font_size, centroid.x, centroid.y, 15, current_centroid_text);

            for (let bound of addedTextBounds) {
              if (rectIntersectsRect(current_zoom_text_bound, bound)) {
                return
              }
            }
            addedTextBounds.add(current_zoom_text_bound);
            
            if (!centroid.current_zoom_text) {
                centroid.current_zoom_text = new PIXI.BitmapText(current_centroid_text, {
                    fontFamily: 'Arial',
                    fontSize: current_centroid_font_size,
                    fontName: "TopicFont",
                    align: 'left',
                    visible: true,
                });
                centroid.current_zoom_text.zIndex = 100;

                // Position the text at the centroid of the cluster
                centroid.current_zoom_text.position.set(centroid.x, centroid.y);
                centroid.current_zoom_text.anchor.set(0.5, 0);

                // Add the text to the viewport
                viewport.addChild(centroid.current_zoom_text);
            } else {
                centroid.current_zoom_text.fontSize = current_centroid_font_size;
                centroid.current_zoom_text.visible = true;
            }
          } 
        });
      }     

      // Adding paper nodes circles to viewport by leaf cluster
      leafClusters.forEach(cluster => {
        let contentSet = new Set(cluster.content);
        let leafClusterNodes = nodes.filter(node => contentSet.has(node.paperId));

        leafClusterNodes.forEach((node, i) => {  
          // Handling Node text, draw labels
          const lambda = (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
          const circleHeight = 1 + (min_font_size / 3) * lambda;
          if(!node.circle) {
              node.circle = new PIXI.Graphics();
              node.circle.zIndex = 55;
              // node.circle.beginFill(0xb9f2ff);
              node.circle.beginFill(circleColorDiamondSequence[cluster.cluster_id % circleColorDiamondSequence.length]);
              node.circle.drawCircle(node.x, node.y, circleHeight);
              node.circleHeight = circleHeight;
              node.circle.endFill();
              viewport.addChild(node.circle);
          } else {
              node.circle.visible = true;
          }
        });
      })

      // Adding paper text labels to viewport by leaf cluster
      leafClusters.forEach(cluster => {
        let contentSet = new Set(cluster.content);
        let leafClusterNodes = nodes.filter(node => contentSet.has(node.paperId));

        leafClusterNodes.forEach((node, i) => {  
          // Handling Node text, draw labels
          const lambda = (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
          const fontSize = (min_font_size + (max_font_size - min_font_size) * lambda / 3);
          let multilineTitle = multilineText(node.title, 30)

          // Not allowing more than 10 paper labels / a lot of words
          if (addedTextBounds.length > 10) {
            return
          }

          // Check for overlaps with existing labels
          let current_zoom_text_bound = labelBounds(fontSize, node.x, node.y, 30, multilineTitle);
          for (let bound of addedTextBounds) {
            if (rectIntersectsRect(current_zoom_text_bound, bound)) {
              return
            }
          }
          addedTextBounds.add(current_zoom_text_bound);

          if(!node.text) {
              node.text = new PIXI.BitmapText(multilineTitle, {
                fontFamily: 'Arial',
                fontSize: fontSize,
                fontName: "TitleFont",
                fill: 0xffffff,
                align: 'left',
                visible: true,
              });
              node.text.zIndex = 60;
              node.text.anchor.set(0.5, 0);
              node.text.position.set(node.x + node.circleHeight, node.y + node.circleHeight + 1);
              viewport.addChild(node.text);
          } else {
              node.text.fontSize = fontSize;
              node.text.visible = true; // make it visible if it already exists
          }
        });
      })
    }

    // Update visibility of circles and text based on the current field of view and zoom level
    let totalUpdateTime = 0
    let numUpdates = 0
    let prev_viewport_bounds = PIXI.Rectangle.EMPTY;
    const updateNodes = () => {
      // Start the timer
      const t0 = performance.now();
      if (!paperNodes) return;

      // get the current field of view
      const viewport_bounds = viewport.getVisibleBounds();
			viewport_bounds.pad(viewport_bounds.width * 0.2);

      // Update visibility of nodes and labels
      if (
        prev_viewport_bounds.x !== viewport_bounds.x ||
        prev_viewport_bounds.y !== viewport_bounds.y ||
        prev_viewport_bounds.width !== viewport_bounds.width ||
        prev_viewport_bounds.height !== viewport_bounds.height
      ) {
        // reset all nodes and labels graphics not in viewport (resetting text globally was messing up the preventing text overlap and deteching text.visible)
        leafClusters.forEach((node, i) => {
          if (node.region) { polygonContainer.removeChild(node.region); };
        })
        clusterCentroids.forEach((centroid, key) => {
          if (centroid.current_zoom_text) { centroid.current_zoom_text.visible = false; };
        })
        paperNodes.forEach((node, i) => {
          if (node.circle) { node.circle.visible = false; };
          if (node.text) { node.text.visible = false; };
          if (node.graphics) { node.graphics.visible = false; };
        })
        let vis_nodes = paperNodes.filter((node) =>
          viewport_bounds.contains(node.x, node.y)
        )
        let vis_cluster_centroids = new Map();
        clusterCentroids.forEach((centroid, key) => {
          if (viewport_bounds.contains(centroid.x, centroid.y)) {
            vis_cluster_centroids.set(key, centroid);
          }
        });

        // Take the top visible nodes
        vis_nodes.sort((a, b) => {
          return b.citationCount - a.citationCount;
        });
        // vis_nodes = vis_nodes.slice(0, 25);

        prev_viewport_bounds = viewport_bounds.clone(); // clone the rectangle to avoid reference issues
        drawNodes(vis_nodes, vis_cluster_centroids, viewport);
      }

      // Performance debugger: Stop the timer and print the time taken, 15 ms is the threshold for smooth animation (60 fps)
      if (logging) {
        const t1 = performance.now();
        const updateTime = t1 - t0;

        if (numUpdates % 60 === 0) {
          totalUpdateTime = 0
          numUpdates = 0
        }

        numUpdates += 1
        totalUpdateTime += updateTime

        console.log("Update time (ms): " + Math.round((t1 - t0)), "avg: ", Math.round(totalUpdateTime / numUpdates));
      }
    };

    // Update nodes based on ticker
    app.ticker.add(updateNodes)

  }, [papersData, edgesData, clusterData]);

  return <div className="pixiContainer" style={{ display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;
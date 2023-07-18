import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { rectIntersectsRect, flattenClusters, multilineText, labelBounds, traverseCluster, calculateClusterCentroids, getVoronoiNeighbors, getColorForClass } from './util';
import { computeHierarchicalLayout } from './layout';

const ResearchPaperPlot = ({ papersData, edgesData, clusterData }) => {
  const pixiContainer = useRef();
  PIXI.BitmapFont.from("TitleFont", { fill: 0xFFFFFF }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });
  PIXI.BitmapFont.from("TopicFont", { fill: 0xFFFFFF, fontWeight: 'bold', }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });

  useEffect(() => {
    const logging = false;
    console.log("papersData", papersData, "edgesData", edgesData, "clusterData", clusterData)

    // Compute force-directed layout of PaperNodes
    let paperNodes = papersData
      .map(({title, x, y, citationCount, paperId, abstract, classification_ids}) => ({title, x: x, y: y, citationCount, paperId, abstract, classification_ids, main_class: classification_ids[0]}))
    let leafClusters = flattenClusters(clusterData);
    let centroidNodes = []

    // hierarchical layout
    const layout = computeHierarchicalLayout(clusterData, paperNodes, edgesData)
    const layoutNodes = layout.nodes;
    const layoutLinks = layout.links;

    console.log("LAYOUT NODES: ", layoutNodes)
    calculateClusterCentroids(leafClusters, layoutNodes, centroidNodes)
    console.log("leaf clusters", leafClusters) // expect 58 leaf clusters bc there are 58 categories
    console.log("centroidNodes", centroidNodes) // expect 58 centroid ndoes for each cluster
    // const edges = layout.edgesData;
    // const centerNodes = layout.centerNodes; // clusterNode!
    const normalizedRadius = layout.normalizedRadius; // this should be used to determine the zoom, currently 230, prev 150
    const zoomScale = normalizedRadius / 300;

    const app = new PIXI.Application({
      width: window.innerWidth,
      height: window.innerHeight - 1000,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
      backgroundColor: 0x5D4D3E,
      resizeTo: window,
    });

    if (pixiContainer.current && pixiContainer.current.childNodes.length > 0) {
      pixiContainer.current.replaceChild(app.view, pixiContainer.current.childNodes[0]);
    } else {
      pixiContainer.current.appendChild(app.view);
    }

    // For now, make the world square and even for better circle parsing
    const minWorldSize = Math.min(window.innerWidth, window.innerHeight);
    const viewport = new Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldWidth: minWorldSize,
      worldHeight: minWorldSize,
      ticker: app.ticker,
      events: app.renderer.events,
      stopPropagation: true,
    });
    viewport.sortableChildren = true;
    viewport.drag().pinch().wheel().decelerate()
    // .clampZoom({ minWidth: 50, maxHeight: viewport.worldHeight / zoomScale, maxWidth: viewport.worldWidth / zoomScale})
    .setZoom(zoomScale)
    .moveCenter(viewport.worldWidth / 2, viewport.worldHeight / 2)
    // viewport.clamp({direction: 'all'})
    app.stage.addChild(viewport);

    // Creates a map from cluster_id to main_topic
    const clusterMap = new Map();
    clusterData.forEach(cluster => traverseCluster(cluster, clusterMap));

    console.log("layoutNodes", layoutNodes)
    const min_scale = Math.min(...layoutNodes.filter(({data}) => data.citationCount !== undefined).map((node) => Math.sqrt(node.data.citationCount)));
    const max_scale = Math.max(...layoutNodes.filter(({data}) => data.citationCount !== undefined).map((node) => Math.sqrt(node.data.citationCount)));
    
    // Calculating voronoi from true centroids of leaf clusters
    const extendFactor = 100 // hardcoding for circle design
    const delaunay = Delaunay.from(centroidNodes.map((node) => [node.x, node.y]));
    const minX = Math.min(...layoutNodes.map((paper) => paper.x));
    const maxX = Math.max(...layoutNodes.map((paper) => paper.x));
    const minY = Math.min(...layoutNodes.map((paper) => paper.y));
    const maxY = Math.max(...layoutNodes.map((paper) => paper.y));
    const voronoi = delaunay.voronoi([minX - extendFactor, minY - extendFactor, maxX + extendFactor, maxY + extendFactor]);

    // scale the data to fit within the worldWidth and worldHeight
    const scaleX = (d) => ((d - minX) / (maxX - minX)) * viewport.worldWidth;
    const scaleY = (d) => ((d - minY) / (maxY - minY)) * viewport.worldHeight;

    // Adding a circle mask and an outline around the circle mask
    let farthestDistance = 0;
    layoutNodes.forEach(node => {
      let distance = Math.sqrt(Math.pow(scaleX(node.x) - viewport.worldWidth / 2, 2) + Math.pow(scaleY(node.y) - viewport.worldHeight / 2, 2));
      if (distance > farthestDistance) {
        farthestDistance = distance;
      }
    });
    let circleMask = new PIXI.Graphics();
    circleMask.beginFill(0x000000); // You can fill with any color
    circleMask.drawCircle(viewport.worldWidth / 2, viewport.worldHeight / 2, farthestDistance + 10);
    circleMask.endFill();
    let outline = new PIXI.Graphics();
    outline.beginFill(0x998D76); // Choose the color for your border
    outline.drawCircle(viewport.worldWidth / 2, viewport.worldHeight / 2, farthestDistance + 15); // +15 instead of +10 to make it larger than the mask
    outline.beginHole();
    outline.drawCircle(viewport.worldWidth / 2, viewport.worldHeight / 2, farthestDistance + 10); // The size of your circle mask
    outline.endHole();
    outline.endFill();
    viewport.addChild(outline)
    const polygonContainer = new PIXI.Container();
    viewport.addChild(polygonContainer);
    polygonContainer.mask = circleMask;
    viewport.addChild(circleMask);
    

    // // Ensuring parentIds extend to the farthest zoom
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

    // // Creates a map from parentId to nodes with that parentId to then calculate centroid for each cluster weighted by paper location
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
    const clusterCentroids = new Map();
    for (let [[clusterId, zoomLevel], nodes] of clusterGroups) {
        let sumX = 0, sumY = 0, totalWeight = 0;
        for (let node of nodes) {
            const weight = node.children.length;
            sumX += node.centroid_x * weight;
            sumY += node.centroid_y * weight;
            totalWeight += weight;
        }
        const key = [clusterId, zoomLevel].toString();
        if (totalWeight > 0) {  // To avoid division by zero
            clusterCentroids.set(key, {x: sumX / totalWeight, y: sumY / totalWeight, layer: zoomLevel});
        }
    }

    // Map cluster_id to classification_id
    let clusterToClassId = new Map();
    leafClusters.forEach(node => {
      clusterToClassId.set(node.id, node.classification_id);
    })
    
    // Hardcoding (zoomLayers) a parent cluster mapping for voronois coloring
    let classColorMap = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
      // Setting parent cluster colors by cluster_id
      leafClusters.forEach(node => {
        let parentId = node.parents[zoomLevel];
        if (parentId !== undefined) {
          let classId = clusterToClassId.get(parentId);
          if (!classColorMap.has(classId)) {
            classColorMap.set(classId, getColorForClass(classId));
          }
        }
      });
    }

    // Sort paperNodes by citationCount to prioritize showing higher citationCount papers
    layoutNodes.sort((a, b) => b.data.citationCount - a.data.citationCount);

    // Scale all coordinates appropriately
    layoutNodes.forEach(node => { node.x = scaleX(node.x); node.y = scaleY(node.y); });
    clusterCentroids.forEach(node => { node.x = scaleX(node.x); node.y = scaleY(node.y); });
    const scaledVoronoi = Array.from(voronoi.cellPolygons()).map(cellPolygon => {
      return cellPolygon.map(point => {
          return [scaleX(point[0]), scaleY(point[1])];
      });
  });

    const drawNodes = (nodes, vis_cluster_centroids, viewport, vis_links) => {
      let zoomLevelAbsolute = ((viewport.scaled) - 2).toFixed(1)
      let zoomLevelAbsoluteFloor = Math.floor(zoomLevelAbsolute)
      let zoomLevel = Math.max(-1, zoomLevelAbsoluteFloor)
      let zoomDecimalToNextZoom = zoomLevelAbsoluteFloor >= -1 ? zoomLevelAbsolute - zoomLevelAbsoluteFloor : 0;
      let originalZoomLevel = zoomLevel;

      // Font size
      const bounds = viewport.getVisibleBounds();
      let min_font_size = bounds.width < bounds.height
          ? bounds.width / (28.3)
          : bounds.height / (50);
      let max_font_size = min_font_size * 1.7;

      const addedTextBounds = new Set();

      // Preview Zoom: Adding cluster polygons on the preview layer to the viewport
      const addClusterPolygons = (node, i, opacity, currentLevel, regionLevel) => {
        // You want the polygon of the parents, not the current node which would be node.parents[currentLevel]
        let parentId = node.parents[currentLevel + 1];
        if (parentId === undefined) {
          parentId = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
        }
        let parentClassId = clusterToClassId.get(parentId)
        let fillColor = classColorMap.get(parentClassId);
        const region = scaledVoronoi[i];

        // For high level zoom colors
        if (regionLevel === 0) {
          // Check if the next parent is the same as the current parent to prevent unnecessary repeated opacity changes
          let parentIdAfter = node.parents[currentLevel + 2];
          if (parentIdAfter === undefined) {
            parentIdAfter = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
          }
          let parentClassId2 = clusterToClassId.get(parentIdAfter)
          if (parentClassId === parentClassId2) {
            opacity = 1
          }

          if (!node.region1) {
            const polygon = new PIXI.Graphics();
            polygon.zIndex = 50;

            polygon.beginFill(fillColor, opacity * 0.7);
            polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
            polygon.endFill();

            node.region1 = polygon;
            node.region1.visible = true;
            polygonContainer.addChild(polygon);
          } else {
            node.region1.clear(); // clear the old drawing
            node.region1.beginFill(fillColor, opacity * 0.7);
            node.region1.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
            node.region1.endFill();
            node.region1.visible = true;
          } 
        }

        // For preview zoom colors
        if (regionLevel === 1) {
          // Check if the previous parent is the same as the current parent to prevent unnecessary repeated opacity changes
          if (currentLevel >= 1) {
            let parentIdBefore = node.parents[currentLevel - 1];
            if (parentIdBefore === undefined) {
              parentIdBefore = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
            }
            let parentClassId2 = clusterToClassId.get(parentIdBefore)
            if (parentClassId === parentClassId2) {
              opacity = 0
            }
          }

          if (!node.region2) {
            const polygon = new PIXI.Graphics();
            polygon.zIndex = 50;

            polygon.beginFill(fillColor, opacity * 0.7);
            polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
            polygon.endFill();

            node.region2 = polygon;
            node.region2.visible = true;
            polygonContainer.addChild(polygon);
          } else {
            node.region2.clear(); // clear the old drawing
            node.region2.beginFill(fillColor, opacity * 0.7);
            node.region2.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
            node.region2.endFill();
            node.region2.visible = true;
          } 
        }
      }
      leafClusters.forEach((node, i) => {
        addClusterPolygons(node, i, 1 - zoomDecimalToNextZoom, zoomLevel, 0)
        addClusterPolygons(node, i, zoomDecimalToNextZoom, zoomLevel + 1, 1)
      });

      // Current Zoom: Adding the cluster text to viewport, and any in the next 3 layers
      for (let i = 0; i < 3; ++i) {
        vis_cluster_centroids.forEach((centroid, key) => {
          let [clusterId] = key.split(',').map(Number); 
          if (centroid.layer === zoomLevel + i) {
            let topCategory = "Unknown";
            if(clusterMap.has(clusterId)){
                topCategory = clusterMap.get(clusterId);
            }

            // Create new text
            let current_centroid_font_size = max_font_size / (1.5 ** (centroid.layer - originalZoomLevel));
            let multilineSize = 15;
            let current_centroid_text = multilineText(topCategory, multilineSize);

            // Check for font size bounds
            // if (current_centroid_font_size < min_font_size) {
            //   return
            // }

            // Not allowing more than 20 labels
            if (addedTextBounds.size > 20) {
              return
            }

            // Check for overlaps with existing labels
            let current_zoom_text_bound = labelBounds(current_centroid_font_size, centroid.x, centroid.y, multilineSize, current_centroid_text);

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
                centroid.current_zoom_text.anchor.set(0.5, 0.5); // center it on the centroid

                // Add the text to the viewport
                viewport.addChild(centroid.current_zoom_text);
            } else {
                centroid.current_zoom_text.fontSize = current_centroid_font_size;
                centroid.current_zoom_text.visible = true;
            }
          } 
        });
      }     

      // Adding paper nodes to viewport by leaf cluster: 1 ms
      nodes.forEach(node => {
        // Add circles to viewport
        let type = node['data'].children ? "cluster" : "paper"
        if (type === "cluster") { return }

        const lambda = (Math.sqrt(node.data.citationCount) - min_scale) / (max_scale - min_scale);
        let fontSize = max_font_size * 0.8
        const multilineSize = 30
        fontSize = min_font_size + (max_font_size - min_font_size) * (lambda / 3);
        const circleHeight = 1 + (max_font_size - min_font_size) * (lambda / 100);

        if(!node.circle) {
            node.circle = new PIXI.Graphics();
            node.circle.zIndex = 55;
            if (type === "cluster") {
              node.circle.beginFill(0x808080);
            } else {
              node.circle.beginFill(0xF5F5F0, 1);
            }
            
            node.circle.drawCircle(node.x, node.y, circleHeight);
            node.circleHeight = circleHeight;
            node.circle.endFill();
            viewport.addChild(node.circle);
        } else {
            node.circle.visible = true;
        }

        // Add the text to the viewport
        
        
        let multilineTitle = multilineText(node['data'].name, multilineSize)

        // Not allowing more than 20 paper labels / a lot of words
        if (addedTextBounds.size > 20) {
          return
        }

        // Check for overlaps with existing labels
        let current_zoom_text_bound = labelBounds(fontSize, node.x, node.y, multilineSize, multilineTitle);
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
                fill: 0xFFFBF1,
                align: 'left',
                visible: true,
              });
            node.text.zIndex = 60;
            node.text.anchor.set(0.5, 0);
            node.text.position.set(node.x + node.circleHeight, node.y + node.circleHeight);
            viewport.addChild(node.text);
        } else {
            node.text.fontSize = fontSize;
            node.text.visible = true; // make it visible if it already exists
        }
      });

      // Add layout edges between nodes
      vis_links.forEach(edge => {
        // optimize this later!
          // console.log(edge.source)
          const sourceNode = layoutNodes.find(node => node === edge.source);
          const targetNode = layoutNodes.find(node => node === edge.target);

          // hide topic connecting edges
          if (sourceNode.children || targetNode.children) {
            return
          }
      
          // Create a new graphics object for the edge if it doesn't exist
          if (!edge.edge_graphics) {
            edge.edge_graphics = new PIXI.Graphics();
            edge.edge_graphics.zIndex = 50; // set this below node's zIndex to ensure nodes are drawn on top
            viewport.addChild(edge.edge_graphics);

            // Draw the line
            edge.edge_graphics.clear(); // remove any existing line
            edge.edge_graphics.lineStyle(1, 0x808080, 0.5 + edge.weight / 2); // set the line style (you can customize this)
            edge.edge_graphics.moveTo(sourceNode.x, sourceNode.y); // move to the source node's position
            edge.edge_graphics.lineTo(targetNode.x, targetNode.y); // draw a line to the target node's position
            viewport.addChild(edge.edge_graphics)
          } else {
            edge.edge_graphics.visible = true;  
          }
      });

      // No for loops: 1-2 ms (max is 15 ms)
    }

    // Update visibility of circles and text based on the current field of view and zoom level
    let totalUpdateTime = 0
    let numUpdates = 0
    let prev_viewport_bounds = PIXI.Rectangle.EMPTY;
    let count = 0
    const updateNodes = () => {
      // Start the timer
      const t0 = performance.now();
      // if (!layoutNodes || count > 0) return;

      // get the current field of view
      const viewport_bounds = viewport.getVisibleBounds();
			viewport_bounds.pad(viewport_bounds.width * 0.1);

      // Update visibility of nodes and labels
      if (
        prev_viewport_bounds.x !== viewport_bounds.x ||
        prev_viewport_bounds.y !== viewport_bounds.y ||
        prev_viewport_bounds.width !== viewport_bounds.width ||
        prev_viewport_bounds.height !== viewport_bounds.height
      ) {
        // reset all nodes and labels graphics not in viewport (resetting text globally was messing up the preventing text overlap and deteching text.visible)

        // leafClusters.forEach((node, i) => {
        //   if (node.region1) { node.region1.visible = false; };
        //   if (node.region2) { node.region2.visible = false; };
        // })
        clusterCentroids.forEach((centroid, key) => {
          if (centroid.current_zoom_text) { centroid.current_zoom_text.visible = false; };
        })
        layoutNodes.forEach((node, i) => {
          if (node.circle) { node.circle.visible = false; };
          if (node.text) { node.text.visible = false; };
        })
        layoutLinks.forEach((edge, i) => {
          if (edge.edge_graphics) { edge.edge_graphics.visible = false; }
        })

        let vis_nodes = layoutNodes.filter((node) => viewport_bounds.contains(node.x, node.y))
        let vis_cluster_centroids = new Map();
        clusterCentroids.forEach((centroid, key) => {
          if (viewport_bounds.contains(centroid.x, centroid.y)) {
            vis_cluster_centroids.set(key, centroid);
          }
        });

        // Take the top visible nodes (for performance?)
        vis_nodes = vis_nodes.slice(0, 1000);
        let vis_nodes_set = new Set(vis_nodes);
        let vis_links = layoutLinks.filter((edge) => vis_nodes_set.has(edge.source) && vis_nodes_set.has(edge.target));

        prev_viewport_bounds = viewport_bounds.clone(); // clone the rectangle to avoid reference issues
        // drawNodes(vis_nodes, vis_cluster_centroids, viewport);
        drawNodes(vis_nodes, vis_cluster_centroids, viewport, vis_links);

        count += 1
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

        if (Math.round(totalUpdateTime / numUpdates) > 0) {
          console.log("Update time (ms): " + Math.round((t1 - t0)), "avg: ", Math.round(totalUpdateTime / numUpdates));
        }
      }
    };

    // Update nodes based on ticker
    // updateNodes() // for debugging
    app.ticker.add(updateNodes)

  }, [papersData, edgesData, clusterData]);

  return <div className="pixiContainer" style={{ display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;
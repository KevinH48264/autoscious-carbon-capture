import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { randomDarkModeColor, rectIntersectsRect, sortPoints, getLeafClusters, flattenClusters } from './util';

const ResearchPaperPlot = ({ papersData, topicsData, clusterData }) => {
  const pixiContainer = useRef();

  PIXI.BitmapFont.from("TitleFont", {
    fill: 0xF8F8F8,
    fontSize: 80,
  }, {
    chars: PIXI.BitmapFont.ASCII.concat(['∀']),
  });

  PIXI.BitmapFont.from("TopicFont", {
    fill: 0x000000,
    fontSize: 80,
  }, {
    chars: PIXI.BitmapFont.ASCII.concat(['∀']),
  });

  useEffect(() => {
    fetch('output_100_tsne.json')
      .then(response => response.json())
      .then(json => {
        papersData = json;
        console.log("Papers manual data:", json);
      }).then(() => {

    console.log("PAPERSDATA", papersData, topicsData, clusterData)
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
      .clampZoom({ minWidth: 50, maxHeight: viewport.worldHeight, maxWidth: viewport.worldWidth})
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

    // Creating voronoi from leaf nodes
    let leafClusters = getLeafClusters(clusterData);
    const delaunay = Delaunay.from(leafClusters.map((node) => [scaleX(node.centroid_x), scaleY(node.centroid_y)]));
    const voronoi = delaunay.voronoi([0, 0, viewport.worldWidth, viewport.worldHeight]);

    // This is code for extracting category data into leaf clusters from paperIds
    const papersDataMap = papersData.reduce((acc, paper) => {
      acc[paper.paperId] = paper;
      return acc;
    }, {});

    leafClusters.forEach(cluster => {
      const paperIds = cluster.content;

      // Get papers for this cluster from papersData.
      const papersForCluster = paperIds.map(paperId => papersDataMap[paperId]);

      // Extract categories from the papers.
      const categories = papersForCluster.flatMap(paper => 
        paper.s2FieldsOfStudy.map(field => field.category)
      );

      // Count categories.
      const categoryCounts = categories.reduce((acc, category) => {
        acc[category] = (acc[category] || 0) + 1;
        return acc;
      }, {});

      // Add categories to the cluster.
      cluster.categories = categoryCounts;
    });

    console.log("LEAFCLUSTERSSSS", leafClusters)
    console.log("papers data", papersData)

    


    // Generate a color sequence
    function generateColorSequence(length) {
      let colorSequence = [];
      for (let i = 0; i < length; i++) {
          colorSequence.push(PIXI.utils.string2hex(randomDarkModeColor()));
      }
      return colorSequence;
    }
    const colorSequence = generateColorSequence(301);

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, viewport) => {
      // let zoomLevel = (viewport.scaled - 1) * 100;
      
      let zoomLevel = (viewport.scaled - 0.95) * 8
      // console.log("zoom level 2", zoomLevel)
      zoomLevel = Math.round(zoomLevel)

      let parentColorMap = new Map();
      const parentCategoryMap = new Map();
      const leafClusterCategoryMap = new Map();

      if (!leafClusters[0] || !nodes[0]) {
        return
      }

      leafClusters.forEach(node => {
        const parentId = node.parents[zoomLevel];

        // If there is a parent ID, aggregate categories
        if (parentId) {
          if (!parentColorMap.has(parentId)) {
              parentColorMap.set(parentId, colorSequence[parentId % 301]);
          }

          // this is for topics
          // If the map already contains the parentId, aggregate categories
          if (parentCategoryMap.has(parentId)) {
            const existingCategories = parentCategoryMap.get(parentId);

            // Merge the categories
            Object.keys(node.categories).forEach(category => {
              existingCategories[category] = (existingCategories[category] || 0) + node.categories[category];
            });

          } else {
            // If the map doesn't contain the parentId, add a new entry
            parentCategoryMap.set(parentId, { ...node.categories });
          }
        } else {
          // If there's no parent, just take the top category of the node
          let topCategory = Object.keys(node.categories).reduce((a, b) => node.categories[a] > node.categories[b] ? a : b);
          leafClusterCategoryMap.set(node.cluster_id, topCategory);
        }
      });

       // Now, you can retrieve the top category for each parent
      parentCategoryMap.forEach((categories, parentId) => {
        let topCategory = Object.keys(categories).reduce((a, b) => categories[a] > categories[b] ? a : b);
        // console.log(`Top category for parent ${parentId}: ${topCategory}`);
      });

      // // And the top category for each leafCluster without a parent
      leafClusterCategoryMap.forEach((topCategory, clusterId) => {
        console.log(`Top category for leafCluster ${clusterId}: ${topCategory}`);
      });

      // Obtaining top category: Iterate through leaf clusters again
      leafClusters.forEach(node => {
        let topCategory;

        // Check if node has a parent
        const parentId = node.parents[zoomLevel];
        if (parentId) {
          // Retrieve categories of the parent
          const parentCategories = parentCategoryMap.get(parentId);

          // Find the top category of the parent
          topCategory = Object.keys(parentCategories).reduce((a, b) => parentCategories[a] > parentCategories[b] ? a : b);
        } else {
          // If there's no parent, get the top category of the node itself
          topCategory = leafClusterCategoryMap.get(node.cluster_id);
        }
      })

            
      if (leafClusters) {
        leafClusters.forEach((node, i) => {
          // if (node.region && node.layer !== zoomLevel && !(node.layer < zoomLevel && node.cluster_id in leafClusterIds)) {
          //   node.region.visible = false;
          // } else {

            const parentId = node.parents[zoomLevel];
            let fillColor = colorSequence[node.cluster_id % 301]
            if (parentId) {
              fillColor = parentColorMap.get(parentId);
            }

            if (node.region) {
              node.region.clear()
            }

            // if(!node.region) {
              // console.log("node", node)
              const region = voronoi.cellPolygon(i);
              const lineWidth = 2;
              const lineColor = 0x000000; // black
              const polygon = new PIXI.Graphics();

              polygon.beginFill(fillColor, 0.5);
              polygon.lineStyle(lineWidth, lineColor); // Add this line to draw the border
              polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
              polygon.endFill();

              node.region = polygon;
              viewport.addChild(polygon);

              // Create text for top category
              let topCategory = "undefined";
              if(parentId){
                topCategory = Object.keys(parentCategoryMap.get(parentId)).reduce((a, b) => parentCategoryMap.get(parentId)[a] > parentCategoryMap.get(parentId)[b] ? a : b);
              }
              else{
                topCategory = leafClusterCategoryMap.get(node.cluster_id);
              }

              // Check if node.text already exists
              if (node.text) {
                viewport.removeChild(node.text);
              }

              // Create new text
              node.text = new PIXI.BitmapText(topCategory, {
                fontFamily: 'Arial',
                fontSize: 24,
                fontName: "TopicFont",
                fill: 0x000000,
                align: 'left',
                visible: true,
                zIndex: 10,
              });
              
              // Position the text at the centroid of the region
              node.text.position.set(scaleX(node.centroid_x), scaleY(node.centroid_y));

              node.text.anchor.set(0.5, 0);

              // Add the text to the viewport
              viewport.addChild(node.text);
            // } else {
            //   node.region.fillColor = fillColor;
            //   node.region.visible = true; // make it visible if it already exists
            // }
          // }
          
        });
      }

      // Handling papers

      // display in order of popularity without overlap
      const bounds = viewport.getVisibleBounds();
      let min_font_size = bounds.width < bounds.height
          ? bounds.width / 17
          : bounds.height / 30;
      const max_font_size = min_font_size * 1.7;
      const min_scale = Math.min(...nodes.map((node) => Math.sqrt(node.citationCount))) - 0.0001;
      const max_scale = Math.max(...nodes.map((node) => Math.sqrt(node.citationCount)));

      nodes.forEach((node, i) => {
        // Handling Node text, draw labels
        const lambda = (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
        const fontSize = min_font_size + (max_font_size - min_font_size) * lambda;
        const circleHeight = 1 + 4 * lambda;

        // if(!node.circle) {
            node.circle = new PIXI.Graphics();
            node.circle.beginFill(0xb9f2ff);
            node.circle.drawCircle(scaleX(node.x), scaleY(node.y), circleHeight);
            node.circle.endFill();
            viewport.addChild(node.circle);
        // } else {
        //     node.circle.visible = true; // make it visible if it already exists
        // }

        // if(!node.text) {
            // TODO: multiline, this can be done in preprocessing
            let words = node.title.split(' ');
            let lines = [];
            let currentLine = '';
            
            for (let word of words) {
                if ((currentLine + ' ' + word).length > 30) {
                    lines.push(currentLine);
                    currentLine = word;
                } else {
                    currentLine += ' ' + word;
                }
            }
            lines.push(currentLine);
            
            let multilineTitle = lines.join('\n').trim();
          
            
            node.text = new PIXI.BitmapText(multilineTitle, {
              fontFamily: 'Arial',
              fontSize: fontSize,
              fontName: "TitleFont",
              fill: 0xffffff,
              align: 'left',
              visible: true,
              zIndex: 10,
            });
            node.text.position.set(scaleX(node.x) + circleHeight, scaleY(node.y) + circleHeight);
            node.text.anchor.set(0.5, 0);
            viewport.addChild(node.text);
        // } else {
        //     node.text.fontSize = fontSize;
        //     node.text.visible = true; // make it visible if it already exists

        //     // Remove overlap between text, I think getBounds can get approximated if it's too slow
            const node_bound = node.text.getBounds();
            for (let j = 0; j < i; j++) {
                const other = nodes[j];
                if (other.text.visible && rectIntersectsRect(node_bound, other.text.getBounds())) {
                    node.text.visible = false;
                    break;
                }
            }
        // }
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
      vis_nodes = vis_nodes.slice(0, 20);

      // Update visibility of nodes and labels
      drawNodes(vis_nodes, viewport);
    };

    // Update nodes based on ticker
    app.ticker.add(updateNodes)

    })
  }, [papersData, topicsData, clusterData]);

  return <div className="pixiContainer" style={{ display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;

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
import requests
import json
import time
import pandas as pd
from pandas.io.json import json_normalize
import folium
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import colors
import warnings
from adjustText import adjust_text
import asyncio
from io import StringIO
import time

#Basic Functions Class

class Helper:
    def coordinate(self,way):
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"[out:json];way({way});node(w);out;"
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            response = response.json()
            flat=flon=count=0
            for i in (response['elements']):
                flat+=i['lat']
                flon+=i['lon']
                count+=1
            1/count
        except Exception as e:
            print("Error",way,e)
            return (None, None)
        return (flat/count, flon/count)
    
    def boundary(self,relation):
        boundaryset = []
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"[out:json];rel({relation});out;"
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            response = response.json()
            for i in (response['elements'][0]['members']):
                if(i['type']=='way' and i['role']=='outer'):
                    coord = self.coordinate(i['ref'])
                    if(coord[0]):
                        boundaryset.append(coord)
                time.sleep(1)
        except:
            print("Error", relation)
            return None
        return boundaryset
    
    def mapcoordinate(self,coordinates,name):
        lattot = 0
        longtot = 0
        numcoord = len(coordinates)
        for i in coordinates:
            lattot += i[0]
            longtot += i[1]
        m = folium.Map(location=[lattot/numcoord, longtot/numcoord], zoom_start=12)
        for coord in coordinates:
            folium.Marker(coord).add_to(m)
        m.save(name + '.html')
        print("Ran Successfully")
        
    def polystring(self,boundarylist):
        string = ""
        for i in boundarylist:
            string+=str(i[0]) + " "
            string+=str(i[1]) + " "
        return string[:-1]
    def markers(self, boundary):
        npb = np.array(boundary)
        xc = npb[:,1]
        yc = npb[:,0]
        tr = [np.max(xc),np.max(yc)]
        tl = [np.min(xc),np.max(yc)]
        br = [np.max(xc),np.min(yc)]
        bl = [np.min(xc),np.min(yc)]
        markers = [list(tr),list(tl),list(br),list(bl)]
        return markers

#Cleaning and querying Data Class

class Clean:
    def boundarydata(self,boundarylist,setting):
        boundarystring = Helper().polystring(boundarylist)
        overpass_url = "http://overpass-api.de/api/interpreter"
        if(setting==0):
            overpass_query = f"""
            [out:csv(::"type", ::"id", ::"lat", ::"lon", name,amenity, place, natural, landuse, 
            brand, shop, building, healthcare; True; ",")]; node(poly:'{boundarystring}');out;
            """
        elif(setting==1):
            overpass_query = f"""
            [out:csv(::"type", ::"id", name, highway, "tiger:name_base", "tiger:name_type", "NHD:FTYPE", natural, 
            waterway, landuse, "addr:county", brand, shop, historic, leisure, building, amenity, man_made, 
            healthcare; True; ",")];way(poly:'{boundarystring}');
            out;
            """
            query2 = f"[out:json];way(poly:'{boundarystring}');out;"
        elif(setting==2):
            overpass_query = f"""
            [out:csv(::"type", ::"id", name, boundary, leisure, landuse, route, place; True; ",")];
            relation(poly:'{boundarystring}');
            out;
            """
            query3 = f"[out:json];relation['name'](poly:'{boundarystring}');out;"
        else:
            print("Invalid Setting")
            return None
        try:
            response = requests.get(overpass_url, params={'data': overpass_query},timeout = 60)
            csv_data = response.text
            df = pd.read_csv(StringIO(csv_data), sep=',')
            if(setting==0):
                df.columns = ["type", "id", "lat", "lon", "tags.name", "tags.amenity",
             "tags.place", "tags.natural", "tags.landuse", "tags.brand",
              "tags.shop","tags.building","tags.healthcare"]
            if(setting==1):
                df.columns = ["type","id", "tags.name", "tags.highway", "tags.tiger:name_base",
                                "tags.tiger:name_type", "tags.NHD:FTYPE", "tags.natural", "tags.waterway", 
                                "tags.landuse", "tags.addr:county", "tags.brand", "tags.shop", "tags.historic", 
                                "tags.leisure", "tags.building","tags.amenity","tags.man_made","tags.healthcare"]
                response = requests.get(overpass_url, params={'data': query2})
                data = response.json()
                waydict = {}
                for element in data['elements']:
                    if 'nodes' in element and 'id' in element:
                        waydict[element['id']] = element['nodes']
                df['nodes'] = df['id'].apply(lambda x: waydict.get(x, []))
            if(setting==2):
                df.columns = ["type","id", "tags.name", "tags.boundary",
                                "tags.leisure", "tags.landuse", "tags.route", "tags.place"]
                response = requests.get(overpass_url, params={'data': query3})
                data = response.json()
                relationdict = {}
                for element in data['elements']:
                    if 'members' in element and 'id' in element:
                        relationdict[element['id']] = element['members']
                df['members'] = df['id'].apply(lambda x: relationdict.get(x, []))
        except Exception as e:
            print(e)
            return None
        return df  
        
    def cleaneddata(self,boundary):
        ways = self.boundarydata(boundary, 1)
        ways['lat'] = None
        ways['lon'] = None
        nodes = self.boundarydata(boundary, 0)
        node_dict = nodes.set_index('id').to_dict(orient='index')
        ways['lat'] = [np.mean([node_dict[node_id]['lat'] for node_id in node_list if node_id in node_dict]) 
                        if any(node_id in node_dict for node_id in node_list) else None
                        for node_list in ways['nodes']]

        ways['lon'] = [np.mean([node_dict[node_id]['lon'] for node_id in node_list if node_id in node_dict]) 
                        if any(node_id in node_dict for node_id in node_list) else None
                        for node_list in ways['nodes']]
        relations = self.boundarydata(boundary, 2)
        node_dict = nodes.set_index('id')[['lat', 'lon']].to_dict(orient='index')
        way_dict = ways.set_index('id')[['lat', 'lon']].to_dict(orient='index')
        lat_list = []
        lon_list = []
        for row in relations.itertuples():
            lat = lon = count = 0
            for member in row.members:
                location = None
                if member['type'] == 'node':
                    location = node_dict.get(member['ref'])
                elif member['type'] == 'way':
                    location = way_dict.get(member['ref'])
                if location:
                    lat += location['lat']
                    lon += location['lon']
                    count += 1
            if count > 0:
                lat_list.append(lat / count)
                lon_list.append(lon / count)
            else:
                lat_list.append(np.nan)
                lon_list.append(np.nan)
        relations['lat'] = lat_list
        relations['lon'] = lon_list
        namedways = (ways[~(ways['tags.name'].isna() | ways['lat'].isna())])
        namedways.drop('nodes', axis=1, inplace=True)
        namednodes = (nodes[~(nodes['tags.name'].isna() | nodes['lat'].isna())])
        namedrelations = (relations[~(relations['tags.name'].isna() | relations['lat'].isna())])
        namedrelations.drop('members', axis=1, inplace=True)
        return namednodes, namedways, namedrelations
    
    def categorizedf(self,df):
        required_columns = ['members', 'tags.tiger:name_type', 'tags.waterway', 'tags.leisure', 
                            'tags.landuse', 'tags.NHD:FTYPE', 'tags.natural', 'tags.tiger:name_base', 
                            'tags.man_made', 'tags.building', 'type', 'tags.name', 'tags.shop', 'tags.brand', 
                            'id', 'tags.place', 'nodes', 'tags.addr:county', 'lat', 'tags.amenity', 'tags.boundary', 
                            'tags.highway', 'tags.route', 'lon', 'tags.historic','tags.healthcare']
        for col in required_columns:
                if col not in df:
                    df[col] = None
        columns = ["Name", "X", "Y", "Label"]
        final = pd.DataFrame(columns=columns)
        label = None
        for index, row in df.iterrows():
            if(row["tags.amenity"] in ["bar","biergarten","cafe","fast_food","food_court","ice_cream","pub",
            "restaurant", "baking_oven", "internet_cafe","kitchen"] or row["tags.building"] in ["bakehouse"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 0}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["college","dancing_school","driving_school","kindergarten","language_school","library", "surf_school"
                    "toy_library","research_institute","training","music_school","school","traffic park","university"] or
                    row["tags.building"] in ["college","kindergarten","school","university"] or
                    row["tags.landuse"] in ["education"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 1}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["bicycle_parking","bicycle_repair_station","bicycle_rental","boat_rental","boat_sharing",
                "bus_station","car_rental","car_sharing","car_wash","compressed_air","vehicle_inspection",
                "charging_station","driver_training","ferry_terminal","fuel","grit_bin","motorcycle_parking",
                "parking","parking_entrance","parking_space","taxi"] or row["tags.building"] in ["train_station","transportation"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 2}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["atm","bank","bureau_de_change"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 3}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["baby_hatch","clinic","dentist","doctors","hospital","nursing_home",
                "pharmacy","social_facility","veterinary", "childcare"] or row["tags.building"] in ["hospital"]
                  or type(row["tags.healthcare"])==str):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 4}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["brothel","casino","gambling","love_hotel","nightclub","stripclub","swingerclub"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 5}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["arts_centre","cinema","community_centre","conference_centre","events_venue","exhibition_centre",
                "fountain","music_venue","planetarium","public_bookcase","social_centre","studio","theatre"] or row["tags.building"]
                  in ["civic","museum","grandstand","pavilion","riding_hall","sports_hall","stadium"] or type(row["tags.leisure"])==str
                  or row["tags.landuse"] in ["fairground","recreation_ground","winter_sports"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 6}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["courthouse","fire_station","police","prison"] or row["tags.building"] in
               ["fire_station","military"] or row["tags.landuse"] in ["military"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 7}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["post_box","post_depot","post_office","ranger_station","townhall"] or 
              row["tags.building"] in ["government","public"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 8}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["animal_boarding","animal_breeding","animal_shelter","animal_training", "clock", "crematorium", "dive_centre",
                "funeral_hall","grave_yard","hunting_stand","kneipp_water_cure","lounger","photo_booth","place_of_mourning",
                "public_bath","refugee_site","vending_machine"] or row["tags.building"] in ["bridge","gatehouse","toilets","hangar","hut",
                "shed","carport","garage","garages","parking","digester","service","transformer_tower","water_tower",
                "storage_tank","silo","construction","container","roof","ruins","yes"] or type(row["tags.man_made"])==str or
                  row["tags.landuse"] in ["brownfield","cemetery","conservation","greenfield","landfill","village_green"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 9}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["monastery","place_of_worship"] or row["tags.building"] in ["cathedral","chapel","church","kingdom_hall",
                "monastery","mosque","presbytery","religious","shrine","synagogue","temple"] or row["tags.landuse"] in ["religious"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 10}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(type(row["tags.highway"])==str):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 11}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.building"] in ["bungalow","cabin","detached","farm","ger","house","houseboat","residential",
                "static_caravan", "stilt_house","beach_hut","tent"] or row["tags.landuse"] in ["residential"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 12}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.building"] in ["apartments","barracks","dormitory","hotel","semidetached_house","terrace","bunker"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 13}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.building"] in ["castle","tower"] or type(row["tags.historic"])==str or row["tags.boundary"] in ["marker"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 14}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.landuse"] in ["forest","meadow","orchard","grass"] or (type(row["tags.natural"])==str and row["tags.natural"]
                not in ["bay","beach","blowhole","cape","coastline","crevasse","geyser","glacier","hot_spring", "isthmus","mud","reef",
                "shoal","spring","strait","water","wetland"]) or row["tags.boundary"] in ["forest","forest_compartment","national_park","protected_area"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 15}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.landuse"] in ["basin","reservoir","salt_pond"] or type(row["tags.waterway"])==str or row["tags.natural"]
               in ["bay","beach","blowhole","cape","coastline","crevasse","geyser","glacier","hot_spring","isthmus","mud",
                "reef","shoal","spring","strait","water","wetland"] or type(row["tags.NHD:FTYPE"])==str):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 16}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(type(row["tags.route"])==str):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 17}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.place"] in ["neighbourhood", "city_block","plot","farm"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 18}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue
            if(row["tags.amenity"] in ["marketplace"] or row["tags.building"] in ["commercial","industrial","kiosk","office","retail",
                "supermarket","warehouse","barn","conservatory","cowshed","farm_auxiliary","greenhouse","slurry_tank","stable","sty","livestock"]
                  or type(row["tags.shop"])==str or type(row["tags.brand"])==str or row["tags.landuse"] in ["commercial", "construction","industrial","retail","institutional",
                "aquaculture","allotments","farmland","farmyard","paddy","animal_keeping","greenhouse_horticulture",
                "plant_nursery","vineyard","depot","garages","port","quarry","railway"]):
                new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 19}
                new_row_df = pd.DataFrame([new_row_data], columns=columns)
                final = pd.concat([final, new_row_df], ignore_index=True)
                continue      
            new_row_data = {"Name": row["tags.name"],"X": row["lat"],"Y": row["lon"],"Label": 20}
            new_row_df = pd.DataFrame([new_row_data], columns=columns)
            final = pd.concat([final, new_row_df], ignore_index=True)
        return final
    
    def fullycleanedf(self,boundary):
        n, w, r = self.cleaneddata(boundary)
        clean_n = self.categorizedf(n)
        clean_w = self.categorizedf(w)
        clean_r = self.categorizedf(r)
        cleaned_df = pd.concat([clean_n, clean_w, clean_r], ignore_index=True)
        return cleaned_df
    
    def wordparse(self, df, name,mincat):
        df = df[~((df['Label'] == 11) & (df.duplicated(subset=['Name', 'Label'])))]
        df = df[df['Label'] != 20]

        # Read the list of top 10000 most common English words
        with open('CommonWords.txt', 'r') as f:
            all_common_english_words = set(f.read().splitlines())

        # Initialize dictionaries to store the frequency of each root word across distinct categories and names
        all_root_word_frequency = defaultdict(set)
        all_root_word_count = defaultdict(int)

        # Tokenize the names and associate root words and their combinations with categories
        for index, row in df.iterrows():
            name, category = row['Name'], row['Label']
            tokens = name.split()


            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens) + 1):
                    root_word = ' '.join(tokens[i:j])
                    all_root_word_frequency[root_word].add(category)
                    all_root_word_count[root_word] += 1


        # Sort, filter, and format
        sorted_all_root_words = sorted(all_root_word_frequency.items(), key=lambda x: (len(x[1]), len(x[0])), reverse=True)
        filtered_root_words = [(root_word, categories) for root_word, categories in sorted_all_root_words if root_word.lower() not in all_common_english_words and len(root_word) > 1]
        filtered_result = [(root_word, len(categories), all_root_word_count[root_word]) for root_word, categories in filtered_root_words]

        # Create DataFrame
        filtered_df = pd.DataFrame(filtered_result, columns=['Root Word', 'Number of Categories', 'Number of Occurrences'])

        # Filter rows based on character conditions
        filtered_df = filtered_df[
            (filtered_df['Root Word'].str.len() > 2) & 
            (~filtered_df['Root Word'].str.contains('[&-]')) &
            (~filtered_df['Root Word'].str.contains('\d'))
        ]
        filtered_df['Number of Occurrences'] = filtered_df['Root Word'].apply(lambda x: all_root_word_count[x])

        # Filter out rows with 'Number of Categories' < 3
        filtered_df = filtered_df[filtered_df['Number of Categories'] >= mincat]

        # Sort by the length of 'Root Word'
        filtered_df['length'] = filtered_df['Root Word'].apply(len)
        filtered_df = filtered_df.sort_values('length', ascending=False).drop('length', axis=1).reset_index(drop=True)

        # Initialize a list to hold the rows to keep
        rows_to_keep = []

        for i, row in filtered_df.iterrows():
            root1 = row['Root Word']
            count1 = row['Number of Categories']
            occurrences1 = row['Number of Occurrences']
            categories1 = all_root_word_frequency[root1]

            if ' ' not in root1:  # Only consider one-word roots for removal
                should_remove = False
                for j, row2 in filtered_df.iterrows():
                    root2 = row2['Root Word']
                    count2 = row2['Number of Categories']
                    if root1 in root2.split() and root1 != root2:
                        categories2 = all_root_word_frequency[root2]
                        if len(categories2) / len(categories1) >= 0.8:
                            should_remove = True
                            break
                if not should_remove:
                    rows_to_keep.append((root1, count1, occurrences1))
            else:
                rows_to_keep.append((root1, count1, occurrences1))

        # Create the final DataFrame from the rows to keep
        final_filtered_df = pd.DataFrame(rows_to_keep, columns=['Root Word', 'Number of Categories', 'Number of Occurrences'])

        # Sort by 'Number of Categories' and 'Root Word'
        final_filtered_df = final_filtered_df.sort_values(by=['Number of Categories', 'Root Word'], ascending=[False, True])

        # Additional filtering based on character conditions
        final_filtered_df = final_filtered_df[(final_filtered_df['Root Word'].str.len() > 2) & (~final_filtered_df['Root Word'].str.contains('[&-]'))].reset_index(drop=True)
        
        # Rank the rows based on 'Number of Categories' and 'Number of Occurrences'
        final_filtered_df['Cat_Rank'] = final_filtered_df['Number of Categories'].rank(method='min', ascending=False)
        final_filtered_df['Occ_Rank'] = final_filtered_df['Number of Occurrences'].rank(method='min', ascending=False)

        # Calculate the 'Score' based on the ranks
        final_filtered_df['Score'] = final_filtered_df['Cat_Rank'] * final_filtered_df['Occ_Rank']

        # Drop the temporary rank columns
        final_filtered_df.drop(['Cat_Rank', 'Occ_Rank'], axis=1, inplace=True)

        # Initialize a list to hold the rows to remove based on the 10% rule
        rows_to_remove = set()

        for i, row1 in final_filtered_df.iterrows():
            root1, occurrences1 = row1['Root Word'], row1['Number of Occurrences']
            if ' ' not in root1:  # Only consider one-word roots for comparison
                for j, row2 in final_filtered_df.iterrows():
                    root2, occurrences2 = row2['Root Word'], row2['Number of Occurrences']
                    if root1 in root2.split() and root1 != root2:
                        if occurrences2 / occurrences1 < 0.10:
                            rows_to_remove.add(root2)

        # Remove the rows that violate the 10% rule
        final_filtered_df = final_filtered_df[~final_filtered_df['Root Word'].isin(rows_to_remove)].reset_index(drop=True)

        # List of prepositions to filter out
        prepositions = ['of', 'at', 'in', 'by', 'for', 'with', 'on', 'to', 'about', 'against']

        # Remove rows starting with a preposition or with three characters
        final_filtered_df = final_filtered_df[~final_filtered_df['Root Word'].str.startswith(tuple(prepositions))]
        final_filtered_df = final_filtered_df[~(final_filtered_df['Root Word'].str.len() == 3)]

        # Reset index after filtering
        final_filtered_df.reset_index(drop=True, inplace=True)
        
        file_path = name + ".csv"
        final_filtered_df.to_csv(file_path, index=False)
        
        return final_filtered_df

#Visualization Class
class Visualize:
    def cluster_and_process(self, data, eps, cleaned):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X_scaled)
        clusters_dict = {}
        for i, label in enumerate(labels):
            #print(cleaned)
            try:
                node_name = [str(cleaned.iloc[i]['Name']),str(cleaned.iloc[i]['Label']),str(cleaned.iloc[i]['X']),str(cleaned.iloc[i]['Y'])]
            except:
                continue
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(node_name)

        return labels, clusters_dict
    
    def cluster(self, df,phrase,epsilon,display, markers):
        mask = df['Name'].str.contains(phrase)
        clean_mask = mask.fillna(False)
        cleaned = df[clean_mask]
        cleaned = cleaned[~((cleaned['Label'] == 11) & (cleaned.duplicated(subset=['Name', 'Label'])))]
        X = cleaned[['X', 'Y']].values
        additional_points = np.array(markers)
        additional_df = pd.DataFrame(additional_points, columns=['X', 'Y'])
        df = pd.concat([df, additional_df], ignore_index=True)
        labels, clusters_dict = self.cluster_and_process(X, epsilon, cleaned)
        if(display):
            plt.scatter(X[:, 1], X[:, 0], c=labels, cmap='viridis', s=50)  
            plt.scatter(additional_points[:,0], additional_points[:,1], color='red', s=50)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('DBSCAN Clustering')
            plt.show()
            for cluster_label, names in clusters_dict.items():
                print(f"Cluster {cluster_label}: {names}")
        if(not display):
            return clusters_dict.items()
    def dictionary(self, filtered,unique, eps, markers):
        finaldict = {}
        df = filtered
        for word in df['Root Word']:
            try:
                clusters = Visualize().cluster(unique, word,eps,False, markers)
            except:
                pass
            maxscore = 0
            bestcluster = None
            for i in clusters:
                if(i[0] == -1):
                    continue
                typeset = set()
                for j in i[1]:
                    typeset.add(int(j[1]))
                if(len(typeset)>=4 and len(typeset)>maxscore):
                    maxscore = len(typeset)
                    bestcluster = i
            if(bestcluster and len(bestcluster)>1):
                finaldict[word] = bestcluster[1]
            else:
                finaldict[word] = None
        return finaldict
    
    def visualizecities(self,data, minp, maxp, markers):
        custom_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', 
                     '#FFC0CB', '#800080', '#FFFF00', '#00FFFF', 
                     '#FF4500', '#7CFC00', '#7FFF00', '#32CD32', 
                     '#ADFF2F', '#20B2AA', '#FF1493', '#8A2BE2', 
                     '#40E0D0', '#8B008B', '#FF6347', '#FFD700']
        
        boundary_points = []
        xc = []
        yc = []
        for i in markers:
            xc.append(i[0])
            yc.append(i[1])
            boundary_points.append(tuple(i))

        warnings.filterwarnings("ignore")
        fig, ax = plt.subplots(figsize=(13, 6))
        cmap = colors.ListedColormap(custom_colors)
        norm = Normalize(vmin=0, vmax=len(data.keys()) - 1)

        boundary_x, boundary_y = zip(*boundary_points)
        ax.scatter(boundary_x, boundary_y, marker='x', color='red', label='County Boundary', s=200)

        texts = []

        for i, (key, value) in enumerate(data.items()):
            if value is not None:
                if minp <= len(value[1]) <= maxp:
                    coordinates = [(float(item[3]), float(item[2])) for item in value]
                    x, y = zip(*coordinates)
                    scatter = ax.scatter(x, y, label=key, color=custom_colors[i % 18], s=10)
                    texts.append(ax.text(x[0], y[0], key, fontsize=12, ha='center', va='center'))

        ax.scatter(xc,yc,color='red', marker='o', s=200, label='County Points')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'), ax=ax)

        plt.tight_layout()
        plt.show()
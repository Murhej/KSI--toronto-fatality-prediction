import {MapContainer, TileLayer, GeoJSON} from 'react-leaflet';
import { useEffect, useRef, useState } from 'react';



function TorontoMap({areaType, onAreaSelect }){
    const [geoData, setGeoData] = useState(null);
    const mapRef = useRef();
    

        useEffect(()=>{
            const url= 
            areaType === "Police" 
            ? '/toronto_divisions.geojson' 
            : '/toronto_Neighbourhoods.geojson'
            fetch(url)
            .then(res =>res.json())
            .then(data => setGeoData(data));
        }, [areaType]);



    
    const onEachDivision = (feature, layer)=>{
        layer.on({
            click: ()=>{
                const bounds = layer.getBounds();
                layer._map.fitBounds(bounds);
                let popup= '';
                let selecteCode='';
                if(areaType === 'Police'){
                    selecteCode = feature.properties.UNIT_NAME;
                    popup = `        
                    <strong>Division:</strong> ${feature.properties.UNIT_NAME} <br/>
                    <strong>Police Department:</strong> ${feature.properties.ADDRESS}
                    `;
                }
                else {
                     selecteCode = feature.properties.AREA_SHORT_CODE;

                    popup =`
                    <strong>Hoods # </strong> ${feature.properties.AREA_SHORT_CODE} <br/>
                    <strong>Area Name</strong> ${feature.properties.AREA_NAME}
                    `
                }
                if (onAreaSelect){
                    onAreaSelect(selecteCode);
                }
                layer.bindPopup(popup).openPopup();

            }
        });
    };
    return (
        <MapContainer 
        center={[43.7, -79.4]} 
        zoom={10} 
        style={{height: '100%', width: '100%'}}
        whenCreated={(mapInstance)=>{
            mapRef.current = mapInstance;
        }}
        >
            <TileLayer 
                url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
            />

            {geoData && <GeoJSON data={geoData} onEachFeature={onEachDivision}/>}
        </MapContainer>
    );
}

export default TorontoMap;
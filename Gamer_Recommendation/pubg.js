
const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3003;

const api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiI3YjUzMDdkMC1jOTIyLTAxM2MtY2RjMi0xNjk1N2FkNjZkNGEiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNzEwOTY0ODUzLCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6ImNyZXcifQ.pp70wu_tbsAI8tch3ukfW-vE7gcD7sog8DCJxwRzM58'; // Replace with your actual API key
const platform = 'steam'; // Assuming 'steam' as the platform
const baseURL = 'https://api.pubg.com/shards/';
const playerName = 'SteelStorm'; // The player name you want to fetch

app.get('/', async (req, res) => {
    try {
        const playerInfo = await fetchPlayerInfo(api_key, playerName);
        const accountId = playerInfo.data[0].id;
        const lifetimeStats = await fetchLifetimeStats(api_key, platform, accountId);
        console.log(lifetimeStats);
        console.log("Lifetime Stats:");
        console.log("Season:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['solo'], null, 2));
        console.log("Solo FPP Matches:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['solo-fpp'], null, 2));
        console.log("Duo Matches:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['duo'], null, 2));
        console.log("Duo FPP Matches:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['duo-fpp'], null, 2));
        console.log("Squad Matches:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['squad'], null, 2));
        console.log("Squad FPP Matches:", JSON.stringify(lifetimeStats.data.attributes.gameModeStats['squad-fpp'], null, 2));

        res.redirect('./thank-you');
    } catch (error) {
        console.error(`Error in processing request: ${error}`);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});



app.get('/thank-you', (req, res) => {
    res.send('Thank you for your request!');
});

async function fetchPlayerInfo(api_key, playerName) {
    try {
        const response = await axios.get(`${baseURL}${platform}/players?filter[playerNames]=${playerName}`, {
            headers: {
                'Accept': 'application/vnd.api+json',
                'Authorization': `Bearer ${api_key}`
            }
        });
        return response.data;
    } catch (error) {
        console.error(`Error in fetchPlayerInfo: ${error}`);
        throw error;
    }
}

async function fetchLifetimeStats(api_key, platform, accountId) {
    try {
        const response = await axios.get(`${baseURL}${platform}/players/${accountId}/seasons/lifetime?filter[gamepad]=false`, {
            headers: {
                'Accept': 'application/vnd.api+json',
                'Authorization': `Bearer ${api_key}`
            }
        });
        return response.data;
    } catch (error) {
        console.error(`Error in fetchLifetimeStats: ${error}`);
        throw error;
    }
}

app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});





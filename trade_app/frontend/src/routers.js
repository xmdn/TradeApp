import {createRouter, createWebHistory} from 'vue-router'
import Home from './components/Home'
import Stocks from './components/Stocks'
import StocksDetail from './components/StocksDetail'

const routes = [
    {
        path:'/',
        name:'home',
        component:Home
    },
    {
        path:'/stocks',
        name:'stocks',
        component:Stocks
    },
    {
        path:'/details/:id',
        name:'details',
        component:StocksDetail,
        props:true
    }
]

const router = createRouter({
    history:createWebHistory(),
    routes
})

export default router;
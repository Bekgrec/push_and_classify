local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.PALLETREPOSITORY)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/BlueWorkforce/scripts/palletRepository/common")
                require("/BlueWorkforce/scripts/palletRepository/customization_main")
                require("/BlueWorkforce/scripts/palletRepository/customization_data")
                require("/BlueWorkforce/scripts/palletRepository/customization_ext")
                require("/BlueWorkforce/scripts/palletRepository/customization_dlg")
            end
        else
            -- Child script

        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
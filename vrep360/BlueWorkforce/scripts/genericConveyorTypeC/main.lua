local isCustomizationScript=sim.getScriptAttribute(sim.getScriptHandle(sim.handle_self),sim.scriptattribute_scripttype)==sim.scripttype_customizationscript

if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.CONVEYOR)
        model.conveyorType='C'
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                -- Common:
                require("/BlueWorkforce/scripts/conveyor_common/common")
                require("/BlueWorkforce/scripts/conveyor_common/customization_main")
                require("/BlueWorkforce/scripts/conveyor_common/customization_data")
                require("/BlueWorkforce/scripts/conveyor_common/customization_ext")
                require("/BlueWorkforce/scripts/conveyor_common/customization_dlg")
                -- Type A specific:
                require("/BlueWorkforce/scripts/genericConveyorTypeC/common")
                require("/BlueWorkforce/scripts/genericConveyorTypeC/customization_main")
                require("/BlueWorkforce/scripts/genericConveyorTypeC/customization_dlg")
            end
        else
            -- Child script
            if model.modelVersion==1 then
                -- Common:
                require("/BlueWorkforce/scripts/conveyor_common/common")
                require("/BlueWorkforce/scripts/conveyor_common/child_main")
                -- Type A specific:
                require("/BlueWorkforce/scripts/genericConveyorTypeC/common")
                require("/BlueWorkforce/scripts/genericConveyorTypeC/child_main")
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
